import numpy as np
from numpy.linalg import norm
from numpy import sqrt, abs
from dataclasses import dataclass
from pysces.qcRunners import TeraChem
from pysces.qcRunners.TeraChem import TCRunner, TCJob, TCJobBatch
from pysces.fileIO import LoggerData, H5File, h5py

from . import NumDeriv as numD
import pickle
import os
from copy import deepcopy
import warnings
from typing import Literal
from time import time

try:
    from tcparse import TCParser
except:
    TCParser = None


AU_2_EV = 27.2114079527
EV_2_AU = 1/27.2114079527

BOHR_2_ANG = 0.529177249
ANG_2_BOHR = 1/BOHR_2_ANG

DEBYE_2_AU = 0.3934303
AU_2_DEBYE = 1/DEBYE_2_AU

class AdiabaticStates():
    def __init__(self, n_states, n_nuclei) -> None:
        self._n_states = n_states
        self._n_nuclei = n_nuclei
        self.eigen_vals = np.zeros(n_states)
        self.eigen_vecs = np.zeros((n_states, n_states))
        self._hamiltonian = np.zeros((n_states, n_states))
        self._dH = np.zeros((n_states, n_states, n_nuclei*3))
        self.NACs = np.zeros((n_states, n_states, n_nuclei*3))
        self.eigen_val_gradients = np.zeros((n_states, n_nuclei*3))
        self.eigen_vec_gradients = np.zeros((n_states, n_states, n_nuclei*3))
        self._diagonalized = False

    @property
    def n_states(self): return self._n_states

    @property
    def n_nuclei(self): return self._n_nuclei

    @property
    def hamiltonian(self):
        return self._hamiltonian
        
    @hamiltonian.setter
    def hamiltonian(self, matrix: np.ndarray):
        if matrix.shape != self._hamiltonian.shape:
            raise ValueError(f'Atempting to set Hamiltonian to a size of {matrix.shape} when it should be {self._hamiltonian.shape}')
        self._hamiltonian = matrix
        self._hamiltonian.flags['WRITEABLE'] = False

        #   clear out the eigen arrays since they no longer belong to this Hamiltonian
        for x in (self.eigen_vals, self.dH, self.NACs, self.eigen_val_gradients, self.eigen_vec_gradients):
            x = np.zeros(x.shape)
        self._diagonalized = False

    @property
    def dH(self):
        return self._dH
    
    @dH.setter
    def dH(self, matrix: np.ndarray):
        if matrix.shape != self._dH.shape:
            raise ValueError(f'Atempting to set Hamiltonian gradient to a size of {matrix.shape} when it should be {self._dH.shape}')
        self._dH = matrix
        self._dH.flags['WRITEABLE'] = False

    def diagonalize_H(self, ref_eig_vecs=None, swap_signs=False):
        e_vals, e_vecs = np.linalg.eigh(self._hamiltonian)
        order = np.argsort(e_vals)
        self.eigen_vals = e_vals[order]
        self.eigen_vecs = e_vecs[:, order]
        self.eigen_vecs[:, 0] *= -1

        #   use the convention that the component of each vector with
        #   that is largest in magnitude is always positive
        if swap_signs:
            for i in range(self.eigen_vecs.shape[1]):
                if self.eigen_vecs[np.argmax(np.abs(self.eigen_vecs[:, i])), i] < 0:
                    self.eigen_vecs[:, i] *= -1

        if ref_eig_vecs is None:
            return self.eigen_vals, self.eigen_vecs
    
        #   correct eigen vectors by multiplying by -1 if their angle
        #   with a reference set is greater than 90 degrees
        corrected_evecs = np.zeros_like(e_vecs)
        for i in range(e_vecs.shape[1]):
            dot = np.dot(e_vecs[:, i], ref_eig_vecs[:, i])
            corrected_evecs[:, i] = e_vecs[:, i]*np.sign(dot)
        self.eigen_vecs = corrected_evecs

        self._diagonalized = True

        return self.eigen_vals, self.eigen_vecs
    
    def eigen_value_gradient(self):
        C_mu_i = self.eigen_vecs
        self.eigen_val_gradients = np.einsum('mi,mnk,ni->ik', C_mu_i, self.dH, C_mu_i)
        self.eigen_val_gradients.flags['WRITEABLE'] = False
        return self.eigen_val_gradients
    
    def eigen_vector_gradient(self):
        e_vec_grads = np.zeros((self._n_states, self._n_states, self._n_nuclei*3))
        for s in range(self._n_states):
            for i, E_i in enumerate(self.eigen_vals):
                C_i = self.eigen_vecs[:, i]
                for j, E_j in enumerate(self.eigen_vals):
                    C_j = self.eigen_vecs[:, j]
                    if i == j: continue
                    inverse_energies = 1/(E_i - E_j)
                    C_dH_C = np.einsum('i,ijk,j->k', C_j, self.dH, C_i)
                    # e_vec_grads[:, i] += inverse_energies * C_dH_C * C_j[:, None]

                    for nuc in range(self._n_nuclei*3):
                        e_vec_grads[s, i, nuc] += inverse_energies * np.dot(C_j, np.dot(self.dH[:, :, nuc], C_i))*C_j[s]

        # e_vec_grads = np.transpose(e_vec_grads, [1,0,2])
        self.eigen_vec_gradients = e_vec_grads
        self.eigen_vec_gradients.flags['WRITEABLE'] = False
        return self.eigen_vec_gradients
    
    def NA_coupling(self, basis_NACs):
        #   use this basis NACS in the calcualtion of the coupling in the polariton basis
        couplings = np.zeros((self._n_states, self._n_states, self._n_nuclei*3))
        for i, E_i in enumerate(self.eigen_vals):
            C_i = self.eigen_vecs[:, i]
            for j, E_j in enumerate(self.eigen_vals):
                C_j = self.eigen_vecs[:, j]
                if i == j: continue
                inverse_energies = 1/(E_j - E_i)
                term1 = inverse_energies * np.einsum('m,mnk,n->k', C_i, self.dH, C_j)
                term2 = np.einsum('m,mnk,n->k', C_i, basis_NACs, C_j)
                # term1 = np.zeros_like(dH[i, j])
                # term2 = np.zeros_like(dH[i, j])
                # for k in range(term2.shape[0]):
                #     term1[k] = -inverse_energies*C_i @ dH[:,:,k] @ C_j # THE CORRECT ONE!!!
                #     term2[k] = C_i @ basis_NACs[:,:,k] @ C_j

                couplings[i, j] = term1 + term2

        self.NACs = couplings
        self.NACs.flags['WRITEABLE'] = False
        return self.NACs
    
    def overlap_matrix(self, wfn_j: 'AdiabaticStates', sub_basis_overlap, wfn_i: 'AdiabaticStates'=None):
        if wfn_i is None:
            wfn_i = self

        e_vecs_i = wfn_i.eigen_vecs
        e_vecs_j = wfn_j.eigen_vecs
        
        overlap = np.zeros((self._n_states, self._n_states))
        for i in range(self._n_states):
            C_i = e_vecs_i[:, i]
            for j in range(self._n_states):
                C_j = e_vecs_j[:, j]
                overlap[i, j] = C_i @ sub_basis_overlap @ C_j

        return overlap
    
    def numerical_gradients(self, states: list['AdiabaticStates'], sub_basis_overlaps: list[np.ndarray], n_points=3, dx=0.01):
        n_states = len(states)
        
        
        #   first make sure all states are diagonalized
        for state in states:
            if not state._diagonalized:
                print("NOT DIAGONALIZED")
                state.diagonalize_H()

        for x in (self.dH, self.NACs, self.eigen_val_gradients, self.eigen_vec_gradients):
            x = np.zeros(x.shape)

        self.dH = numD.compute([s.hamiltonian for s in states], n_points, dx, (1, 2, 0))
        self.eigen_val_gradients = numD.compute([s.eigen_vals for s in states], n_points, dx).T
        self.eigen_vec_gradients = numD.compute([s.eigen_vecs for s in states], n_points, dx, (1, 2, 0))
        basis_overlaps = [self.overlap_matrix(states[i], sub_basis_overlaps[i]) for i in range(n_states)]
        self.NACs = numD.compute(basis_overlaps, n_points, dx, (1, 2, 0))
        for i in range(self.n_nuclei*3):
            np.fill_diagonal(self.NACs[:, :, i], 0.0)

class CoupledMolecule(AdiabaticStates):

    def __init__(self, omega_c, n_elec, n_nuc, field_dir=None) -> None:
        super().__init__(n_elec*2, n_nuc)
        self._field_dir = field_dir
        if self._field_dir is not None:
            self._field_dir = field_dir/np.linalg.norm(field_dir)
        self._omega_c = omega_c
        self._gc = None
        self._n_elec = n_elec

        n_pol = 2
        self._n_dim = n_pol * self._n_elec
        self.matter_states = np.zeros(self._n_dim, dtype=int)
        self.polariton_states = np.zeros(self._n_dim, dtype=int)
        self.state_pairs = np.zeros((self._n_dim, 2), dtype=int)
        for n in range(n_pol):
            for a in range(n_elec):
                count = n*n_elec + a
                self.state_pairs[count] = (a, n)
                self.matter_states[count] = a
                self.polariton_states[count] = n

        #   molecular properties
        self.mol_energies = np.zeros(self._n_elec)
        self.mol_gradients = np.zeros((self._n_elec, self.n_nuclei*3)) 
        self.mol_NACs = np.zeros((self._n_elec, self._n_elec, self.n_nuclei*3))
        self.mol_dipole_matrix = np.zeros((self._n_elec, self._n_elec, 3))
        self.mol_dipole_matrix_gradient = np.zeros((self._n_elec, self._n_elec, self.n_nuclei*3, 3))

        # self.hamiltonian = np.zeros((self._n_dim, self._n_dim))
        # self.e_vals = np.zeros(self._n_dim)
        # self.e_vecs = np.zeros((self._n_dim, self._n_dim))
    
    def copy(self):
        new_copy = deepcopy(self)
        return new_copy
    
    def NA_coupling(self, mol_basis_NACs):
        basis_NACs = self.get_basis_NACs(mol_basis_NACs)
        return super().NA_coupling(basis_NACs)

    def set_hamiltonian(self, energies, dipole_matrix):
        '''
            Evaluate the Hamiltonian elements

            Parameters
            ----------
            energies: np.ndarray
                diagonal components of the hamiltonian
            dipole_matrix: np.ndarray (n_states x n_states)
                Dipole matrix with diagonal elemnts being the dipoles of each
                state and the off-diagonal elements being the transition dipoles
        '''
        n_elec = self._n_elec

        mu_dot_field = np.zeros((n_elec, n_elec))
        if self._field_dir is not None:
            norm_field_dir = self._field_dir / np.linalg.norm(self._field_dir)
            for a in range(n_elec):
                for b in range(n_elec):
                    mu_dot_field[a, b] = np.dot(dipole_matrix[a, b], self._field_dir)
        else:
            for a in range(n_elec):
                for b in range(n_elec):
                    mu_dot_field[a, b] = np.linalg.norm(dipole_matrix[a, b])
                    
        dipole_self = np.zeros((n_elec, n_elec))
        for a in range(n_elec):
            for b in range(n_elec):
                for gamma in range(n_elec):
                    dipole_self[a, b] += self.gc**2/self.omega_c * mu_dot_field[a, gamma]*mu_dot_field[gamma, b]

        H_t = np.zeros((self._n_dim, self._n_dim))
        delta = np.eye(self._n_dim)
        for i, state_i in enumerate(self.state_pairs):
            for j, state_j in enumerate(self.state_pairs):
                a, m = state_i
                b, n = state_j

                H_en = energies[a]*delta[a, b]*delta[m, n]
                H_p = self.omega_c*(n + 0/2)*delta[a, b]*delta[m, n]
                H_en_p = self.gc * mu_dot_field[a, b] * (sqrt(n)*delta[m, n-1] + sqrt(n+1)*delta[m, n+1])
                H_d = dipole_self[a, b]*delta[m, n]

                H_t[i, j] = H_en + H_p + H_en_p + H_d

        self.mol_energies = energies
        self.mol_dipole_matrix = dipole_matrix
        self.hamiltonian = H_t
        return H_t
    
    def set_hamiltonian_gradient(self, gradients, dipoles, dipole_grads):
        '''
            Parameters
            ----------
            gradients: (M_states, N_atoms) array
            dipoles: (M_states, M_states, 3) array
            dipole_grads: (M_states, M_states, N_atoms*3, 3) array
        '''

        n_elec = self._n_elec

        mu_dot_field = np.zeros((n_elec, n_elec))
        grad_mu_dot_field = np.zeros((n_elec, n_elec, self.n_nuclei*3))
        dH = np.zeros((self._n_dim, self._n_dim, self.n_nuclei*3))

        #   dipole dotted with electric field directions
        if self._field_dir is not None:
            for a in range(n_elec):
                for b in range(n_elec):
                    mu_dot_field[a, b] = np.dot(dipoles[a, b], self._field_dir)
                    for nuc in range(self.n_nuclei*3):
                        grad_mu_dot_field[a, b, nuc] = np.dot(dipole_grads[a, b, nuc], self._field_dir)
        #   Or, we assume that the field is always in the same direction as the dipole moments 
        else:
            for a in range(n_elec):
                for b in range(n_elec):
                    field = dipoles[a, b]/np.linalg.norm(dipoles[a, b])
                    mu_dot_field[a, b] = np.linalg.norm(dipoles[a, b])
                    for nuc in range(self.n_nuclei*3):
                        grad_mu_dot_field[a, b, nuc] = np.dot(dipole_grads[a, b, nuc], field)
                    
        #   gradient of the dipole self energy
        dipole_self_grad = np.zeros((n_elec, n_elec, self.n_nuclei*3))
        pre_factor = self.gc**2/self.omega_c
        for a in range(n_elec):
            for b in range(n_elec):
                for nuc in range(self.n_nuclei*3):
                    for gamma in range(n_elec):
                        term1 = pre_factor * grad_mu_dot_field[a, gamma, nuc] * mu_dot_field[gamma, b]
                        term2 = pre_factor * mu_dot_field[a, gamma] * grad_mu_dot_field[gamma, b, nuc]
                        dipole_self_grad[a, b, nuc] += term1 + term2


        #   now form the gradient of the Hamiltonian matrix
        delta = np.eye(self._n_dim)
        for i, state_i in enumerate(self.state_pairs):
            for j, state_j in enumerate(self.state_pairs):
                a, m = state_i
                b, n = state_j

                for nuc in range(self.n_nuclei*3):
                    dH_en = gradients[a, nuc]*delta[a, b]*delta[m, n]
                    dH_p = 0.0
                    dH_en_p = self.gc * grad_mu_dot_field[a, b, nuc] * (sqrt(n)*delta[m, n-1] + sqrt(n+1)*delta[m, n+1])
                    dH_d = dipole_self_grad[a, b, nuc]*delta[m, n]

                    dH[i, j, nuc] = dH_en + dH_p + dH_en_p + dH_d

        self.mol_gradients = gradients
        self.mol_dipole_matrix_gradient = dipole_grads
        self.dH = dH
        return dH
    
    def get_basis_overlaps(self, sub_basis_overlaps):
        overlaps = np.zeros((self._n_dim, self._n_dim))
        for i, state_i in enumerate(self.state_pairs):
            for j, state_j in enumerate(self.state_pairs):
                a, m = state_i
                b, n = state_j
                overlaps[i, j] = sub_basis_overlaps[a, b]*(m == n)

        return overlaps
    
    def get_basis_NACs(self, sub_basis_NACs):
        #   first form couplings matrix in the molecule/photon basis
        basis_NACs = np.zeros((self._n_dim, self._n_dim, self._n_nuclei*3))
        for i, state_i in enumerate(self.state_pairs):
            for j, state_j in enumerate(self.state_pairs):
                a, m = state_i
                b, n = state_j
                basis_NACs[i, j] = sub_basis_NACs[a, b]*(m == n)
        return basis_NACs

    def _overlap_matrix(self, e_vecs_i, evecs_j, mol_overlaps):
        #   first form overlap matrix in the molecule/photon basis
        basis_overlap = np.zeros((self._n_dim, self._n_dim))
        for i, state_i in enumerate(self.state_pairs):
            for j, state_j in enumerate(self.state_pairs):
                a, m = state_i
                b, n = state_j
                basis_overlap[i, j] = mol_overlaps[a, b]*(m == n)
        
        #   now use this matrix in the calcualtion of the overlap in the polariton basis
        overlap = np.zeros((self._n_dim, self._n_dim))
        for i, state_i in enumerate(self.state_pairs):
            C_i = e_vecs_i[:, i]
            for j, state_j in enumerate(self.state_pairs):
                C_j = evecs_j[:, j]
                overlap[i, j] = C_i @ basis_overlap @ C_j
        return overlap
    
    @property
    def omega_c(self):
        return self._omega_c
    
    @property
    def gc(self):
        return self._gc

    def set_gc_from_coupling(self, coupling, trans_dipole):
        scale_factor = coupling/(norm(trans_dipole)*sqrt(self.omega_c))
        g_c = scale_factor*np.sqrt(self.omega_c)
        self._gc = g_c

    def compute_adiabatic_states(self, energies, dipoles):

        # gc = _get_gc(coupling, omega_c, trans_dipole)
        self.hamiltonian = self._get_Nstate_hamiltonian(energies, dipoles)
        self.e_vals, self.e_vecs = self.diagonalize_H(self.hamiltonian)

class PolaritonLogger():
    def __init__(self) -> None:
        self._h5_file: H5File = None
        self._h5_group: h5py.Group
        self._initialized = False
        self._next_dataset: dict[str, np.ndarray] = {}
        self._labels: dict[str,list[str]] = {}

    def setup(self, logging_dir: str, h5_file: H5File):
        self._logging_Dir = logging_dir
        self._h5_file = h5_file
        self._h5_group = h5_file.create_group('polariton')
        self._h5_group.create_dataset('time', shape=(0,), maxshape=(None,), chunks=True)

    def _initialize(self):
        for key, data in self._next_dataset.items():
            ds = self._h5_group.create_dataset(key, shape=(0,)+data.shape, maxshape=(None,)+data.shape)
            ds.attrs.create('labels', self._labels)

    def set_labels(self, labels: dict[str,list[str]]):
        self._labels = labels.copy()

    def add_next_dataset(self, data: dict):
        self._next_dataset = data

    def write(self, logger_data: LoggerData):
        if not self._initialized:
            self._initialize()
        H5File.append_dataset(self._h5_group['time'], logger_data.time)
        for key, data in self._next_dataset.items():
            H5File.append_dataset(self._h5_group[key], data)

class TCPolaritonRunner(TCRunner):
    def __init__(self,
                 coupled_mol: CoupledMolecule,
                 hosts: str, 
                 ports: int, 
                 atoms: list,
                 tc_options: dict, 
                 tc_spec_job_opts: dict = {}, 
                 tc_initial_job_options: dict = None, 
                 server_roots='.', 
                 start_new: bool = False, 
                 run_options: dict = {}, 
                 max_wait=20) -> None:
        super().__init__(hosts, ports, atoms, tc_options, tc_spec_job_opts, tc_initial_job_options, server_roots, start_new, run_options, max_wait)

        self.coupled_mol = coupled_mol

        self._spec_job_opts['gradient_0'] = {'dipolederivative': 'yes'}
        self._spec_job_opts['gradient_1'] = {'cistransdipolederiv': 'yes', 'cisdipolederiv': 'yes'}

        self._prev_evecs = None
        self._prev_ref_job = None

        self._logger = PolaritonLogger()

    @property
    def logger(self):
        return self._logger

    def run_new_geom(self, geom):
        mol = self.coupled_mol
    
        #   Run TeraChem
        if not TeraChem._DEBUG:
            job_batch = self.run_TC_new_geom(geom)
        else:
            if not os.path.isfile('_ref_jobs.pkl'):
                job_batch = self.run_TC_new_geom(geom)
                with open('_ref_jobs.pkl', 'wb') as file: 
                    pickle.dump(job_batch, file)
            else:
                with open('_ref_jobs.pkl', 'rb') as file:
                    job_batch = pickle.load(file)
                    self._prev_jobs = job_batch.jobs
        job_batch: TCJobBatch

        #   update jobs form tc.out file, and correct with esp charges
        for job in self._prev_jobs:
            print(job.results.keys())
            self._update_job_from_tcout(job)
            if job.state == 0:
                continue
            if self._prev_ref_job is None:
                self._prev_ref_job = job
            TeraChem._correct_signs_from_charges(job, self._prev_ref_job)

        _, mol.mol_energies, mol.mol_gradients, mol.mol_NACs, _ = TeraChem.format_output_LSCIVR(job_batch.results_list)

        #   dipole and other gradients
        for job in self._prev_jobs:
            if job.name == 'gradient_1':
                mol.mol_dipole_matrix = self.dipole_matrix_from_job(job)
        mol.mol_dipole_matrix_gradient = self.dipole_matrix_gradient_from_jobs(job_batch.jobs)

        #   set hamiltonian, diagonalize, and compute needed gradients
        hamiltonian = mol.set_hamiltonian(mol.mol_energies, mol.mol_dipole_matrix)
        mol.set_hamiltonian_gradient(mol.mol_gradients, mol.mol_dipole_matrix, mol.mol_dipole_matrix_gradient)
        mol.diagonalize_H(ref_eig_vecs=self._prev_evecs)
        mol.NA_coupling(mol.mol_NACs)
        mol.eigen_value_gradient()
        mol.eigen_vector_gradient()
        self._prev_evecs = mol.eigen_vecs

        #   log all computed quantities
        logged_data = {}
        logged_data['hamiltonian'] = mol.hamiltonian
        logged_data['eigenvalues'] = mol.eigen_vals
        logged_data['eigenvectors'] = mol.eigen_vecs
        logged_data['hamiltonian_grads'] = mol.dH
        logged_data['eigenvalue_grads'] = mol.eigen_val_gradients
        logged_data['NACs'] = mol.NACs
        logged_data['dipole_matrix'] = mol.mol_dipole_matrix
        logged_data['dipole_matrix_grads'] = mol.mol_dipole_matrix_gradient
        self.logger.add_next_dataset(logged_data)

        #   timings, all_energies, elecE, grad, nac, trans_dips
        #   TODO: Compute transition dipoles!!!
        return job_batch.timings, mol.eigen_vals, mol.eigen_vals, mol.eigen_val_gradients, mol.NACs, None
        

    def run_numerical_derivatives(self, mol_geom: np.ndarray, n_points=3, dx=0.01, run_overlaps=True, overlaps=None, set_dipoles=True):
        '''
            Parameters
            ----------
            mol_geom: np.ndarray
                molecule geometry to use as the reference point
            run_overlaps: bool
                if True, overlaps will be computed. This can not be done for CIS methods
            overlaps: List or np.ndarray
                Use these overlaps w.r.t. the reference wave functions instead of computing them
        '''

        #   run the reference set of jobs
        mol = self.coupled_mol
        self.run_new_geom(mol_geom)
        # ref_job: TCJob = self._prev_jobs[-1]
        ref_job: TCJob = self._prev_jobs[1]
        self._update_job_from_tcout(ref_job)
        all_energies, ref_energies, grads, nacs, trans_dips = TeraChem.format_output_LSCIVR([x.results for x in self._prev_jobs])


        #   run all of the numerical derivative jobs
        jobs = super()._run_numerical_derivatives(ref_job, n_points, dx, run_overlaps)
        num_deriv_jobs: list[TCJob] = jobs['num_deriv_jobs']
        overlap_jobs: list[TCJob] = jobs['overlap_jobs']

        n_enrgies = len(ref_job.results['energy'])
        if n_enrgies != self.coupled_mol._n_elec:
            raise ValueError(f'Computed electronic structure states ({n_enrgies}) does not equal the Coupled Molecule electronic states ({self.coupled_mol._n_elec})')

        #   update all jobs
        # for num_deriv_job, overlap_job in zip(jobs['num_deriv_jobs'], jobs['overlap_jobs']):
        start_time = time()
        for i, num_deriv_job in enumerate(num_deriv_jobs):
            self._update_job_from_tcout(num_deriv_job)
            if run_overlaps and overlap_jobs is not None:
                self._update_job_from_tcout(overlap_jobs[i])
                TeraChem._correct_signs_from_overlaps(num_deriv_job, overlap_jobs[i])
            else:
                TeraChem._correct_signs_from_charges(num_deriv_job, ref_job)
        # print("TIME: ", time()- start_time)


        #   diagonalize reference hamiltonian, their eigenvectors will be used as a reference
        ref_dipoles = self.dipole_matrix_from_job(ref_job)
        mol.set_hamiltonian(ref_energies, ref_dipoles)
        mol.diagonalize_H()

        #   each coupled AdibaticState is computed for each numerical job
        states = []
        all_dipoles = []
        all_energies = []
        for i in range(len(num_deriv_jobs)):
            coupled = mol.copy()
            energies = num_deriv_jobs[i].results['energy']
            dipoles = self.dipole_matrix_from_job(num_deriv_jobs[i])
            coupled.set_hamiltonian(energies, dipoles)
            coupled.diagonalize_H(mol.eigen_vecs)
            all_energies.append(energies)
            states.append(coupled)
            all_dipoles.append(dipoles)
            
        #   molecular overlaps
        sub_basis_overlaps = []
        if overlaps is not None:
            #   overlap matricies were provided
            sub_basis_overlaps = overlaps.copy()
        elif ref_job.excited_type == 'cis' or not run_overlaps:
            #   cis jobs can't compute overlaps in TeraChem, so we approximate them with the NAC
            print("CIS APPROXIMATION: ", len(num_deriv_jobs))

            for i in range(0, len(num_deriv_jobs), n_points-1):
                shift_multiples = (np.arange(n_points) - n_points//2).tolist()
                shift_multiples.pop(n_points//2)
                for j in shift_multiples:
                    overlap_matrix = nacs[:, :, i//(n_points-1)]*dx*j
                    np.fill_diagonal(overlap_matrix, 1.0)
                    sub_basis_overlaps.append(overlap_matrix)

                # print("COMPARE: ")
                # print(overlaps[i])
                # print(overlap_matrix)
        else:
            #   get the overlaps from the CAS jobs
            for i in range(len(num_deriv_jobs)):
                sub_basis_overlaps.append(overlap_jobs[i].results['ci_overlap'])

        sub_basis_overlaps = np.array(sub_basis_overlaps)

        #   molecule properties and gradients
        mol.mol_gradients = numD.compute(all_energies, n_points, dx).T
        mol.mol_dipole_matrix_gradient = numD.compute(all_dipoles, n_points, dx, (1, 2, 0, 3))
        mol.mol_NACs = numD.compute(sub_basis_overlaps, n_points, dx, (1, 2, 0))
        mol.mol_dipole_matrix = self.dipole_matrix_from_job(ref_job)
        
        #   polariton overlaps
        pol_overlaps = [mol.get_basis_overlaps(x) for x in sub_basis_overlaps]

        mol.numerical_gradients(states, pol_overlaps, n_points, dx)
        print(f'{mol.NACs[:, :, 0]}')

        return jobs, states

    def run_numerical_derivatives_TMP(self, ref_job: TCJob, dx=0.01, excited_method='cas'):
        overlap = (excited_method == 'cas')
        jobs = super()._run_numerical_derivatives(ref_job, dx, overlap)

        for num_deriv_job, overlap_job in zip(jobs['num_deriv_jobs'], jobs['overlap_jobs']):
            self._update_job_from_tcout(num_deriv_job)
            self._update_job_from_tcout(overlap_job)
            if overlap:
                TeraChem._correct_signs_from_overlaps(num_deriv_job, overlap_job)

        return jobs
            

    def _update_job_from_tcout(self, job: TCJob):
        '''
            Append job jets from the tc.out file. This requires the use of the TCParser repo.
        '''
        if TCParser is None:
            warnings.warn('TCParser could not be imported, please install TCParser')
            return

        job_data = TCParser().parse_from_list(job.results['tc.out'])
        for key in job_data:
            if key not in job.results:
                job.results[key] = job_data[key]

    def dipole_matrix_from_job(self, tc_job: TCJob):
        '''
            re-order and combine all of the the dipoles from a TC job dict
            into a single matrix.
        '''

        excited_type = tc_job.excited_type
        if excited_type == 'cis':
            dipole_key = 'cis_unrelaxed_dipoles'
            tr_dipole_key = 'cis_transition_dipoles'
        elif excited_type == 'cas':
            dipole_key = 'cas_dipoles'
            tr_dipole_key = 'cas_transition_dipoles'
        else:
            raise ValueError(f"Invalid job type specified '{excited_type}'; must be 'cis' or 'cas'")

        job_results = tc_job.results
        n_states = len(job_results['energy'])
        dipole_matrix = np.zeros((n_states, n_states, 3))
        dipole_matrix[0, 0] = np.array(job_results['dipole_vector'])*DEBYE_2_AU
        for i in range(1, n_states):
            dipole_matrix[i, i] = job_results[dipole_key][i-1]
        indicies = np.transpose(np.tril_indices(n_states, k=-1))
        for count, (i, j) in enumerate(indicies):
            dipole_matrix[i, j] = job_results[tr_dipole_key][count]
            dipole_matrix[j, i] = job_results[tr_dipole_key][count]

        return dipole_matrix
    

    def dipole_matrix_gradient_from_jobs(self, tc_jobs: list[TCJob]):
        '''
            re-order and combine all of the the dipole derivatives from a list of 
            TC jobs into a single matrix.
        '''
        dipole_grads = np.zeros_like(self.coupled_mol.mol_dipole_matrix_gradient)
        n_states = self.coupled_mol._n_elec
        got_gs, got_ex, got_tr = False, False, False
        for tc_job in tc_jobs:

            if 'cis_transition_dipole_deriv' in tc_job.results:
                derivs = np.array(tc_job.results['cis_transition_dipole_deriv'])
                #   swap second and 4th axis. The last axis is now mX,mY,mZ
                #   then, flatten the middle two axis, which are the cartesian coordinates
                n_elms = n_states*(n_states-1)//2
                derivs = derivs.transpose((0, 2, 3, 1)).reshape(n_elms, -1, 3)

                indicies = np.transpose(np.tril_indices(n_states, k=-1))
                for count, (i, j) in enumerate(indicies):
                    dipole_grads[i, j] = derivs[count]
                    dipole_grads[j, i] = derivs[count]
                got_tr = True

            if 'cis_dipole_deriv' in tc_job.results:
                # for line in tc_job.results['tc.out']:
                #     print(line)
                derivs = np.array(tc_job.results['cis_dipole_deriv'])
                derivs = derivs.transpose((0, 2, 3, 1)).reshape(n_states-1, -1, 3)
                for i in range(1, n_states):
                    dipole_grads[i, i] = derivs[i-1]
                got_ex = True

            if 'dipole_deriv' in tc_job.results:
                derivs = np.array(tc_job.results['dipole_deriv'])
                print(f'{derivs.shape=} {dipole_grads.shape=}')
                derivs = derivs.transpose((1, 2, 0)).reshape(-1, 3)
                dipole_grads[0, 0] = derivs
                got_gs = True

        if not got_gs:
            raise ValueError('Could not recover ground state dipole moment derivatives from TC jobs')
        if not got_ex:
            raise ValueError('Could not recover excited state dipole moment derivatives from TC jobs')
        if not got_tr:
            raise ValueError('Could not recover transition state dipole moment derivatives from TC jobs')

        return dipole_grads

