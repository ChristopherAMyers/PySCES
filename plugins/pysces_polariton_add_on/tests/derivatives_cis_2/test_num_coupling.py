from PySCESPol.Polariton import TCPolaritonRunner, CoupledMolecule
from pysces.qcRunners.TeraChem import format_output_LSCIVR

from qcelemental.models import Molecule
from qcelemental import constants
import numpy as np
import json

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    np.testing.assert_allclose(a, a.T, rtol=rtol, atol=atol)

def check_antisymmetric(a, rtol=1e-05, atol=1e-08):
    np.testing.assert_allclose(a, -a.T, rtol=rtol, atol=atol)

AU_2_EV = 27.2114079527
EV_2_AU = 1/27.2114079527

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
    'convthre': 1E-8,
    'sphericalbasis': 'yes',
    'precision': 'double',
    'threall': 1e-20,
    
    #   TD-DFT
    'cis': 'yes',
    'cisnumstates': 2,
    'cisrestart': 'cis_restart',
    'cisconvtol': 1e-8,

    'cischarges': 'yes',
    'resp': 'yes',
}

ref_coupled = CoupledMolecule(5.59728242*EV_2_AU, 3, len(mol_atoms), [1, 1, 1])
ref_coupled.set_gc_from_coupling(0.3*EV_2_AU, 1.5603)

host = '10.141.0.3'
port = 1234
runner = TCPolaritonRunner(ref_coupled, host, port, mol_atoms, tc_options, run_options={'max_state': 2})
runner._server_root_list = ['/home/cmyers7/data/TC_servers']
result = runner.run_new_geom(mol_geom)

ref_job = runner._prev_jobs[1]
all_energies, ref_energies, ref_gradients, ref_mol_coupling, _ = format_output_LSCIVR([x.results for x in runner._prev_jobs])

#   import job that has the correct overlap matrices
num_overlap_data = json.load(open('num_coupling.json'))
mol_overlaps = np.array([num_overlap_data[str(i+1)]['ci_overlap'] for i in range(len(num_overlap_data) - 1)])

#   run numerical derivatives and keep a copy of the coupled molecule
# jobs, states = runner.run_numerical_derivatives(mol_geom, overlaps=mol_overlaps)
ref_coupled = runner.coupled_mol.copy()
jobs, states = runner.run_numerical_derivatives(mol_geom, run_overlaps=False, overlaps=None, n_points=3, dx=0.005)
tst_coupled = runner.coupled_mol.copy()


#   gather maximum deviation statistics and print results
print_arrays = False
ideal_max_pct_diff = 400.75
error_msg = f'Difference between numerical and analytical gradients is greater than {ideal_max_pct_diff:.2f}%'
np.set_printoptions(suppress=True, linewidth=100)
grads = (('deriv evals', tst_coupled.eigen_val_gradients, ref_coupled.eigen_val_gradients),
         ('mol gradients', tst_coupled.mol_gradients, ref_coupled.mol_gradients),
         )
for label, tst, ref in grads:
    print("\n", label)
    for a in range(tst.shape[0]):
        max_val = np.max(np.abs(tst[a]))
        max_pct_diff = 100*np.max(np.abs(tst[a] - ref[a])/max_val)
        max_idx = np.argmax(np.abs(tst[a] - ref[a])/max_val)
        print(f'    state {a:2d}: {max_pct_diff:.5f} % ')
        if max_pct_diff > ideal_max_pct_diff:
                raise AssertionError(error_msg)
        if print_arrays:
            for i in range(0, ref_coupled.n_nuclei*3):
                star_str = '*' if i == max_idx else ''
                print("{:3d} {:12.8f} {:12.8f}".format(i, tst[a, i], ref[a, i]) + star_str)

print("dipole deriv")
tst = tst_coupled.mol_dipole_matrix_gradient
ref = ref_coupled.mol_dipole_matrix_gradient
for a in range(tst.shape[0]):
    range2 = range(tst.shape[1])
    for b in range2:
        max_val = np.max(np.abs(tst[a, b]))
        max_pct_diff = 100*np.max(np.abs(tst[a, b] - ref[a, b])/max_val)
        max_idx = np.argmax(np.abs(tst[a, b] - ref[a, b])/max_val, axis=0).tolist()

        print(f'    state {a:2d}  {b:2d}: {max_pct_diff:10.5f} %')
        if max_pct_diff > ideal_max_pct_diff:
            raise AssertionError(error_msg)
        if print_arrays:
            for i in range(0, ref_coupled.n_nuclei*3):
                format_str = '{:12.8f} '*len(tst[a, b, i])
                format_str += f'| {format_str}'
                star_str = '*' if i in max_idx else ''
                print(f"{i:3d}" + format_str.format( 
                    *tst[a, b, i], *ref[a, b, i]) + star_str)

grads = (
        ('deriv Ham', tst_coupled.dH, ref_coupled.dH, check_symmetric),
        ('evec grads', tst_coupled.eigen_vec_gradients, ref_coupled.eigen_vec_gradients, lambda x: None),
        ('mol coupling', tst_coupled.mol_NACs, ref_coupled.mol_NACs, check_antisymmetric),
        ('pol coupling', tst_coupled.NACs, ref_coupled.NACs, check_antisymmetric),
        )

for label, tst, ref, symm_chk in grads:
    print("\n", label)
    for i in range(0, ref_coupled.n_nuclei*3):
        # symm_chk(tst[:, :, i], rtol=0.04, atol=1e-04)
        symm_chk(ref[:, :, i])
    for a in range(tst.shape[0]):
        range2 = range(tst.shape[1])
        if symm_chk == check_symmetric or symm_chk == check_antisymmetric:
            range2 = range(a, tst.shape[1])
        for b in range2:
            max_val = np.max(np.abs(tst[a, b]))
            if max_val == 0.0 or (np.abs(tst[a, b]).min() < 1e-8 and np.abs(ref[a, b]).min() < 1e-8):
                max_pct_diff = 0.0
                max_idx = np.int64(-1)
            else:
                max_pct_diff = 100*np.max(np.abs(tst[a, b] - ref[a, b])/max_val)
                max_idx = np.argmax(np.abs(tst[a, b] - ref[a, b])/max_val)

            print(f'    state {a:2d}  {b:2d}: {max_pct_diff:10.5f} %')
            if max_pct_diff > ideal_max_pct_diff:
                raise AssertionError(error_msg)
            if print_arrays:
                for i in range(0, ref_coupled.n_nuclei*3):
                    star_str = '*' if i == max_idx else ''
                    print("{:3d} {:12.8f} {:12.8f} ".format(i, 
                        tst[a, b, i], ref[a, b, i]) + star_str)

