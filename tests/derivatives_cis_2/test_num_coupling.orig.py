from pysces.qcRunners.Polariton import TCPolaritonRunner, CoupledMolecule
from pysces.qcRunners.TeraChem import format_output_LSCIVR

from qcelemental.models import Molecule
from qcelemental import constants
import numpy as np
import json


AU_2_EV = 27.2114079527
EV_2_AU = 1/27.2114079527


DEBYE_2_AU = 0.3934303
AU_2_DEBYE = 1/DEBYE_2_AU

DX = 0.01 # in bohr


def get_corrected_overlap_mat(overlap_mat):
    new_mat = np.zeros_like(overlap_mat)
    for i in range(len(overlap_mat)):
        for j in range(len(overlap_mat)):
            if overlap_mat[i, i] < 0:
                new_mat[j, i] = -1*overlap_mat[j, i]
            else:
                new_mat[j, i] = overlap_mat[j, i]
    return new_mat


mol = Molecule.from_file('malonaldehyde.xyz')
mol_atoms = mol.symbols
mol_geom = mol.geometry*constants.bohr2angstroms
tc_options = {
    'atoms': mol_atoms,
    'method': 'hf',
    'basis': 'sto-3g',
    'charge': 0,
    'spinmult': 1,
    'closed_shell': True,
    'restricted': True,
    'precision': 'mixed',
    'convthre': 1E-6,
    'sphericalbasis': 'yes',
    
    #   TD-DFT
    # 'cis': 'yes',
    # 'cisnumstates': 2,
    # 'cisrestart': 'cis_restart',

    #   CAS-CI
    'casci': 'yes',
    'cassinglets': 3,
    'active': '4',
    'closed': 17,
    # 'directci': 'yes',
    'caswritevecs': 'yes',
    'castarget': 2,
    'cascharges': 'yes'
}

ref_coupled = CoupledMolecule(5.59728242*EV_2_AU, 3, len(mol_atoms), [1, 1, 1])
ref_coupled.set_gc_from_coupling(0.3*EV_2_AU, 1.5603)

host = '10.141.0.3'
port = 1234
runner = TCPolaritonRunner(ref_coupled, host, port, mol_atoms, tc_options, run_options={'max_state': 2})
runner._server_root_list = ['/home/cmyers7/data/TC_servers']
result = runner.run_new_geom(mol_geom)

ref_job = runner._prev_jobs[1]
ref_energies, ref_gradients, ref_mol_coupling, _ = format_output_LSCIVR([x.results for x in runner._prev_jobs])


# jobs = runner._run_numerical_derivatives(ref_job, overlap=True)
jobs = runner.run_numerical_derivatives(mol_geom, excited_method='cas')
num_deriv_jobs = jobs['num_deriv_jobs']
overlap_jobs = jobs['overlap_jobs']

#   diagonalize reference job
n_states = len(ref_energies)
dipoles_ref = runner.dipole_matrix_from_job(ref_job, 'cas')
ham_ref = ref_coupled.set_hamiltonian(ref_energies, dipoles_ref)
ref_evals, ref_evecs = ref_coupled.diagonalize_H()


#   import job that has the correct overlap matrices
num_overlap_data = json.load(open('num_coupling.json'))


#   compute numerical derivatives of all the data
ref_dipole_deriv_matrix =   np.zeros(list(dipoles_ref.shape) + [ref_coupled.n_nuclei*3])
tst_dH =                    np.zeros(list(ham_ref.shape) + [ref_coupled.n_nuclei*3])
tst_d_evals =               np.zeros([ref_coupled._n_dim, ref_coupled.n_nuclei*3])
tst_gradient =              np.zeros([n_states, ref_coupled.n_nuclei*3])
tst_mol_coupling =          np.zeros([n_states, n_states, ref_coupled.n_nuclei*3])
tst_pol_coupling =          np.zeros([ref_coupled._n_dim, ref_coupled._n_dim, ref_coupled.n_nuclei*3])
tst_evec_grads =            np.zeros([ref_coupled._n_dim, ref_coupled._n_dim, ref_coupled.n_nuclei*3])

for i in range(0, len(num_deriv_jobs), 2):
    num_job_n = num_deriv_jobs[i+0]
    num_job_p = num_deriv_jobs[i+1]
    overlap_job_n = overlap_jobs[i+0]
    overlap_job_p = overlap_jobs[i+1]


    energy_n = np.array(num_job_n.results['energy'])
    energy_p = np.array(num_job_p.results['energy'])
    dipoles_n = runner.dipole_matrix_from_job(num_job_n, 'cas')
    dipoles_p = runner.dipole_matrix_from_job(num_job_p, 'cas')

    coupled_n = ref_coupled.copy()
    coupled_p = ref_coupled.copy()

    ham_n = coupled_n.set_hamiltonian(energy_n, dipoles_n)
    ham_p = coupled_p.set_hamiltonian(energy_p, dipoles_p)

    evals_n, evecs_n = coupled_n.diagonalize_H(ref_evecs)
    evals_p, evecs_p = coupled_p.diagonalize_H(ref_evecs)

    tst_gradient[:, i//2] = (energy_p - energy_n)/(2*DX)
    ref_dipole_deriv_matrix[:, :, :, i//2] = (dipoles_p - dipoles_n)/(2*DX)
    tst_dH[:, :, i//2] = (ham_p - ham_n)/(2*DX)
    tst_d_evals[:, i//2] = (evals_p - evals_n)/(2*DX)
    tst_evec_grads[:, :, i//2] = (evecs_p - evecs_n)/(2*DX)

    # mol_overlap_n = overlap_job_n.results['ci_overlap']
    # mol_overlap_p = overlap_job_p.results['ci_overlap']
    mol_overlap_n = np.array(num_overlap_data[str(i+1)]['ci_overlap'])
    mol_overlap_p = np.array(num_overlap_data[str(i+2)]['ci_overlap'])

    #   optional approximation
    if False:
        mol_overlap_p = ref_mol_coupling[:,:,i//2]*DX
        mol_overlap_n = -mol_overlap_p
        mol_overlap_p[np.diag_indices_from(mol_overlap_p)] = 1
        mol_overlap_n[np.diag_indices_from(mol_overlap_n)] = 1


    corrected_Sn = get_corrected_overlap_mat(mol_overlap_n)
    corrected_Sp = get_corrected_overlap_mat(mol_overlap_p)

    nacme = (corrected_Sp - corrected_Sn)/(2*DX)
    nacme_sym = (nacme.T - nacme)/2
    tst_mol_coupling[:, :, i//2] = nacme

    # overlap_mat_n = coupled_n.overlap_matrix(ref_evecs, evecs_n, corrected_Sn)
    # overlap_mat_p = coupled_p.overlap_matrix(ref_evecs, evecs_p, corrected_Sp)

    #   STILL WORKING ON IT
    pol_basis_overlap_n = ref_coupled.get_basis_overlaps(corrected_Sn)
    pol_basis_overlap_p = ref_coupled.get_basis_overlaps(corrected_Sp)
    overlap_mat_n = ref_coupled.overlap_matrix(coupled_n, pol_basis_overlap_n)
    overlap_mat_p = ref_coupled.overlap_matrix(coupled_p, pol_basis_overlap_p)

    tst_pol_coupling[:, :, i//2] = (overlap_mat_p - overlap_mat_n)/(2*DX)

#   form the polariton coupling on the reference geometry
ref_dipole_deriv_matrix = np.transpose(ref_dipole_deriv_matrix, [0,1,3,2])
ham_ref =               ref_coupled.set_hamiltonian(ref_energies, dipoles_ref)
ref_evals, ref_evecs =  ref_coupled.diagonalize_H(ref_evecs)
ref_dH =                ref_coupled.set_hamiltonian_gradient(ref_gradients, dipoles_ref, ref_dipole_deriv_matrix) # TODO: use actual gradients
ref_pol_coupling =      ref_coupled.NA_coupling(ref_mol_coupling)
ref_d_evals =           ref_coupled.eigen_value_gradient()
ref_evec_grads =        ref_coupled.eigen_vector_gradient()

#   gather maximum deviations tatistics and print results
print_arrays = False
np.set_printoptions(suppress=True, linewidth=100)
grads = (('deriv evals', tst_d_evals, ref_d_evals),
         ('mol gradients', tst_gradient, ref_gradients),
         )
for label, tst, ref in grads:
    print("\n", label)
    for a in range(tst.shape[0]):
        max_val = np.max(np.abs(tst[a]))
        max_pct_diff = 100*np.max(np.abs(tst[a] - ref[a])/max_val)
        print(f'    state {a:2d}: {max_pct_diff:.5f} % ')
        if max_pct_diff > 0.5:
                raise AssertionError('Difference between numerical and analytical gradients is greater than 0.5%')
        if print_arrays:
            for i in range(0, ref_coupled.n_nuclei*3):
                print("{:12.8f} {:12.8f}".format(tst[a, i], ref[a, i]))

grads = (
         ('deriv Ham', tst_dH, ref_dH),
         ('mol coupling', tst_mol_coupling, ref_mol_coupling),
         ('pol coupling', tst_pol_coupling, ref_pol_coupling),
         ('evec grads', tst_evec_grads, ref_evec_grads),
         )

for label, tst, ref in grads:
    print("\n", label)
    for a in range(tst.shape[0]):
        for b in range(tst.shape[1]):
            max_val = np.max(np.abs(tst[a, b]))
            if max_val == 0.0 or (np.abs(tst[a, b]).min() < 1e-8 and np.abs(ref[a, b]).min() < 1e-8):
                max_pct_diff = 0.0
                max_idx = -1
            else:
                max_pct_diff = 100*np.max(np.abs(tst[a, b] - ref[a, b])/max_val)
                max_idx = np.argmax(np.abs(tst[a, b] - ref[a, b])/max_val)
            print(f'    state {a:2d}  {b:2d}: {max_pct_diff:10.5f} %')
            if max_pct_diff > 0.5:
                raise AssertionError('Difference between numerical and analytical gradients is greater than 0.5%')
            if print_arrays:
                for i in range(0, ref_coupled.n_nuclei*3):
                    star_str = '*' if i == max_idx else ''
                    print("{:12.8f} {:12.8f} ".format(
                        tst[a, b, i], ref[a, b, i]) + star_str)

