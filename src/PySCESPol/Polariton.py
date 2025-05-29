import numpy as np
from numpy.linalg import norm
from numpy import sqrt, abs
from dataclasses import dataclass
from pysces.qcRunners import TeraChem
from pysces.qcRunners.TeraChem import TCRunner, TCJob, TCJobBatch, TCRunnerOptions, format_combo_job_results
from pysces.fileIO import LoggerData, H5File, h5py, TCJobsLogger
from pysces.subroutines import SignFlipper
from qcelemental.models import Molecule
from qcelemental.periodic_table import periodictable as pt

from . import NumDeriv as numD
import pickle
import os
from copy import deepcopy
import warnings
from typing import Literal
from time import time
import json
from collections import deque

try:
    from tcparse import parse_from_list
except:
    warnings.warn('TCParser could not be imported, please install TCParser')

from pprint import pprint

AU_2_EV = 27.2114079527
EV_2_AU = 1/27.2114079527

BOHR_2_ANG = 0.529177249
ANG_2_BOHR = 1/BOHR_2_ANG

DEBYE_2_AU = 0.3934303
AU_2_DEBYE = 1/DEBYE_2_AU

AMU_2_AU = 1.822888486*10**3

#   helper functions
def _expand_array(small_array: np.ndarray, target_shape: tuple, indices: list | np.ndarray) -> np.ndarray:
    """
    Expand a 3D array to a larger size, placing elements at specified indices.
    
    Parameters:
    -----------
    small_array : np.ndarray
        Original array with shape (n, n, d)
    target_shape : tuple
        Shape of the new array (m, m, d) where m > n
    indices : list
        List of indices where the original array should be placed
        
    Returns:
    --------
    np.ndarray
        New array with specified shape with small_array placed at the specified indices
    """
    if small_array.shape[0] != small_array.shape[1]:
        raise ValueError("The first two dimensions of small_array must be equal.")
    
    # Create new array filled with zeros
    new_array = np.zeros(target_shape)
    
    # Place elements from small_array into new_array at the specified indices
    for old_i, new_i in enumerate(indices):
        for old_j, new_j in enumerate(indices):
            new_array[new_i, new_j, :] = small_array[old_i, old_j, :]
    
    return new_array

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
    
    def calc_eigen_value_gradient(self):
        C_mu_i = self.eigen_vecs
        self.eigen_val_gradients = np.einsum('mi,mnk,ni->ik', C_mu_i, self.dH, C_mu_i)
        self.eigen_val_gradients.flags['WRITEABLE'] = False
        return self.eigen_val_gradients
    
    def calc_eigen_vector_gradient(self):
        '''
            this function is teribly slow. Need to optimize and compare einsum methods
        '''
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
    
    def calc_NA_coupling(self, basis_NACs):
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
        # self.NACs.flags['WRITEABLE'] = False
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

    def __init__(self, omega_c, mol_grads, n_nuc, field_dir=None, rwa=False, dse=True) -> None:
        # n_elec = s_high - s_low + 1
        # if s_high < s_low:
        #     raise ValueError('s_high must be greater than or equal to s_low')

        self.mol_grad_indices = mol_grads
        n_elec = max(mol_grads) + 1

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
        self._state_pairs = np.zeros((self._n_dim, 2), dtype=int)
        for n in range(n_pol):
            for a in range(n_elec):
                count = n*n_elec + a
                self.state_pairs[count] = (a, n)
                self.matter_states[count] = a
                self.polariton_states[count] = n
        self._state_pairs = tuple(tuple(p.tolist()) for p in self._state_pairs)

        #   molecular properties
        self.mol_energies = np.zeros(self._n_elec)
        self.mol_gradients = np.zeros((self._n_elec, self.n_nuclei*3)) 
        self.mol_NACs = np.zeros((self._n_elec, self._n_elec, self.n_nuclei*3))
        self.mol_dipole_matrix = np.zeros((self._n_elec, self._n_elec, 3))
        self.mol_dipole_matrix_gradient = np.zeros((self._n_elec, self._n_elec, self.n_nuclei*3, 3))

        #   internal components used to store the hamiltonian
        self._H_d = np.zeros_like(self._hamiltonian)
        self._H_en_p = np.zeros_like(self._hamiltonian)
        self._H_p = np.zeros_like(self._hamiltonian)
        self._H_en = np.zeros_like(self._hamiltonian)
        self._use_DSE = dse # dipole self energy
        self._use_RWA = rwa # use the rotating wave approximation

    
    def copy(self):
        new_copy = deepcopy(self)
        return new_copy
    
    def compute_all(self, all_energies, grads, nacs, dipole_matrix, dipole_matrix_grads, ref_eig_vecs=None):
        self.mol_energies = all_energies
        self.mol_gradients = grads
        self.mol_NACs = nacs
        self.mol_dipole_matrix_gradient = dipole_matrix_grads
        self.mol_dipole_matrix = dipole_matrix

        #   set hamiltonian, diagonalize, and compute needed gradients
        self.set_hamiltonian(all_energies, self.mol_dipole_matrix)
        self.set_hamiltonian_gradient(self.mol_gradients, self.mol_dipole_matrix, self.mol_dipole_matrix_gradient)
        self.diagonalize_H(ref_eig_vecs)
        self.calc_NA_coupling(self.mol_NACs)
        self.calc_eigen_value_gradient()
        # self.calc_eigen_vector_gradient()


    def calc_NA_coupling(self, mol_basis_NACs):
        basis_NACs = self.get_basis_NACs(mol_basis_NACs)
        return super().calc_NA_coupling(basis_NACs)

    def set_hamiltonian(self, energies, dipole_matrix):
        if not self._use_RWA:
            return self._set_hamiltonian_PF(energies, dipole_matrix)
        else:
            return self._set_hamiltonian_RWA(energies, dipole_matrix)

    def _set_hamiltonian_PF(self, energies, dipole_matrix):
        '''
            Evaluate the Pauli-Ferz Hamiltonian elements

            Parameters
            ----------
            energies: np.ndarray
                diagonal components of the hamiltonian
            dipole_matrix: np.ndarray (n_states x n_states)
                Dipole matrix with diagonal elemnts being the dipoles of each
                state and the off-diagonal elements being the transition dipoles
        '''
        n_elec = self._n_elec
        # for i in range(dipole_matrix.shape[0]):
        #     for j in range(dipole_matrix.shape[1]):
        #         print(f'{i=}, {j=}, {dipole_matrix[i,j]=}')

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
                    
        #   dipole self energy
        dipole_self = np.zeros((n_elec, n_elec))
        if self._use_DSE:
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
                self._H_en[i, j] = H_en
                self._H_p[i, j] = H_p
                self._H_en_p[i, j] = H_en_p
                self._H_d[i, j] = H_d

        self.mol_energies = energies
        self.mol_dipole_matrix = dipole_matrix
        self.hamiltonian = H_t
        return H_t
    
    def _set_hamiltonian_RWA(self, energies, dipole_matrix):
        '''
            Evaluate the Jaynes-Cummings Hamiltonian elements

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
            for a in range(n_elec):
                for b in range(n_elec):
                    mu_dot_field[a, b] = np.dot(dipole_matrix[a, b], self._field_dir)
        else:
            for a in range(n_elec):
                for b in range(n_elec):
                    mu_dot_field[a, b] = np.linalg.norm(dipole_matrix[a, b])

        #   dipole self energy
        dipole_self = np.zeros((n_elec, n_elec))
        if self._use_DSE:
            dipole_self = np.zeros((n_elec, n_elec))
            for a in range(n_elec):
                for b in range(n_elec):
                    for gamma in range(n_elec):
                        dipole_self[a, b] += self.gc**2/self.omega_c * mu_dot_field[a, gamma]*mu_dot_field[gamma, b]
        
        #   re-organize the states by "total" excitation
        # excitations = {}
        # for a, n in self.state_pairs:
        #     total = int((a > 0) + n)
        #     if total not in excitations:
        #         excitations[total] = []
        #     excitations[total].append((a, n))
        
        H_t = np.zeros((self._n_dim, self._n_dim))
        # for pair in excitations.values():
        delta = np.eye(self._n_dim)
        for i, state_i in enumerate(self.state_pairs):
            for j, state_j in enumerate(self.state_pairs):
                a, m = state_i
                b, n = state_j

                H_en = energies[a]*delta[a, b]*delta[m, n]
                H_p = self.omega_c*(n + 0/2)*delta[a, b]*delta[m, n]
                H_d = dipole_self[a, b]*delta[m, n]
                # print(H_d)

                if a < b:
                    H_en_p = self.gc * mu_dot_field[a, b] * sqrt(n+1)*delta[m, n+1]
                elif a > b:
                    H_en_p = self.gc * mu_dot_field[a, b] * sqrt(n)*delta[m, n-1]
                else:
                    H_en_p = 0.0

                H_t[i, j] = H_en + H_p + H_en_p
                self._H_en[i, j] = H_en
                self._H_p[i, j] = H_p
                self._H_en_p[i, j] = H_en_p
                self._H_d[i, j] = H_d

        self.mol_energies = energies
        self.mol_dipole_matrix = dipole_matrix
        self.hamiltonian = H_t
        return H_t
    
    def set_hamiltonian_gradient(self, gradients, dipoles, dipole_grads):
        if not self._use_RWA:
            return self._set_hamiltonian_gradient_PF(gradients, dipoles, dipole_grads)
        else:
            return self._set_hamiltonian_gradient_RWA(gradients, dipoles, dipole_grads)

    def _set_hamiltonian_gradient_PF(self, gradients, dipoles, dipole_grads):
        '''
            Parameters
            ----------
            gradients: (M_states, N_atoms) array
            dipoles: (M_states, M_states, 3) array
            dipole_grads: (M_states, M_states, N_atoms*3, 3) array
        '''

        # full_grads = np.zeros((self._n_elec, self.n_nuclei*3))
        # for i, idx in enumerate(self.mol_grad_indices):
        #     print('assigning grads: ', i, idx)
        #     full_grads[idx] = gradients[i]

        full_grads = np.copy(gradients)

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

        if self._use_DSE:
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
                    dH_en = full_grads[a, nuc]*delta[a, b]*delta[m, n]
                    dH_p = 0.0
                    dH_d = dipole_self_grad[a, b, nuc]*delta[m, n]
                    dH_en_p = self.gc * grad_mu_dot_field[a, b, nuc] * (sqrt(n)*delta[m, n-1] + sqrt(n+1)*delta[m, n+1])

                    dH[i, j, nuc] = dH_en + dH_p + dH_en_p + dH_d


        self.mol_gradients = gradients
        self.mol_dipole_matrix_gradient = dipole_grads
        self.dH = dH
        return dH
    
    def _set_hamiltonian_gradient_RWA(self, gradients, dipoles, dipole_grads):
        '''
            Parameters
            ----------
            gradients: (M_states, N_atoms) array
            dipoles: (M_states, M_states, 3) array
            dipole_grads: (M_states, M_states, N_atoms*3, 3) array
        '''

        # full_grads = np.zeros((self._n_elec, self.n_nuclei*3))
        # for i, idx in enumerate(self.mol_grad_indices):
        #     print('assigning grads: ', i, idx)
        #     full_grads[idx] = gradients[i]

        full_grads = np.copy(gradients)

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

        if self._use_DSE:
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
                    dH_en = full_grads[a, nuc]*delta[a, b]*delta[m, n]
                    dH_p = 0.0
                    dH_d = dipole_self_grad[a, b, nuc]*delta[m, n]

                    if a < b:
                        dH_en_p = self.gc * grad_mu_dot_field[a, b, nuc] * sqrt(n+1)*delta[m, n+1]
                    elif a > b:
                        dH_en_p = self.gc * grad_mu_dot_field[a, b, nuc] * sqrt(n)*delta[m, n-1]
                    else:
                        dH_en_p = 0.0

                    dH_en_p = self.gc * grad_mu_dot_field[a, b, nuc] * (sqrt(n)*delta[m, n-1] + sqrt(n+1)*delta[m, n+1])

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

        #   Leaving this for now, as furutre updates should only use
        #   The dimensions that include electronic structure components

        # full_sub_basis_NACs = np.zeros((self._n_elec, self._n_elec, self.n_nuclei*3))
        # for i, idx in enumerate(self.mol_grad_indices):
        #     for j, idx2 in enumerate(self.mol_grad_indices):
        #         full_sub_basis_NACs[idx, idx2] = sub_basis_NACs[i, j]


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
    
    def get_subset_indices(self, *pairs: tuple[int, int]):
        indices = []
        for p in pairs:
            p = tuple(p)
            if p not in self._state_pairs:
                raise ValueError(f'Pair {p} not in state pairs')
            indices.append(self.state_pairs.index(p))

        ordered = sorted(indices)
        index_pair = np.ix_(ordered, ordered)
        return index_pair
    
    @property
    def tr_indicies(self):
        return np.triu_indices(self.n_states, k=+1)

    @property
    def omega_c(self):
        return self._omega_c
    
    @property
    def gc(self):
        return self._gc

    @property
    def state_pairs(self):
        return self._state_pairs

    def set_gc_from_coupling(self, coupling, trans_dipole):
        scale_factor = coupling/(norm(trans_dipole)*sqrt(self.omega_c))
        g_c = scale_factor*sqrt(self.omega_c)
        self._gc = g_c

    def compute_adiabatic_states(self, energies, dipoles):

        # gc = _get_gc(coupling, omega_c, trans_dipole)
        self.hamiltonian = self._get_Nstate_hamiltonian(energies, dipoles)
        self.e_vals, self.e_vecs = self.diagonalize_H(self.hamiltonian)

class DipoleMatrixTracker:
    def __init__(self, order, history_size, interval=1, name='UNKNOWN'):
        self.order = order
        self.history_size = history_size
        self.interval = interval
        self.history_grads = deque(maxlen=history_size)
        self.history_t = deque(maxlen=history_size)
        self.history_f = deque(maxlen=history_size)
        self.polynomial_coeffs = None
        self.name = name

    def update_history(self, new_t, new_grads, new_f):

        #   before we update, let's get one more extrapolation to check how close we are
        if self.polynomial_coeffs is not None:
            guess = np.polyval(self.polynomial_coeffs, new_t)
            guess = guess.reshape(self.history_grads[0].shape)
            error = np.abs(guess - new_grads)
            max_error = np.max(error)
            rms_error = np.sqrt(np.mean(error**2))
            print(f'Before updating extrapolation history in {self.name} at time {new_t}: ')
            print(f'    {max_error=:.5e}, {rms_error=:.5e}')

        self.history_grads.append(np.array(new_grads))
        self.history_t.append(new_t)
        self.history_f.append(new_f)
        print(f'Updating {self.name} history at time {new_t}')
        if len(self.history_grads) == self.history_size:
            self._fit_polynomial()

    def _fit_polynomial(self):
        # Flatten the matrices and stack them into a 2D array
        stacked_history = np.vstack([np.ravel(matrix) for matrix in self.history_grads])
        # Fit a polynomial to each column of the 2D array
        self.polynomial_coeffs = np.polyfit(self.history_t, stacked_history, self.order)

    def check_if_ready(self):
        return len(self.history_grads) == self.history_size
    
    def check_run_deriv(self, new_time):
        if len(self.history_grads) < self.history_size:
            return True, 'History not long enough'
        elif (1+new_time - len(self.history_grads)) % self.interval == 0:
            return True, 'Interval reached'
        else:
            return False, 'OK'

    def predict_at_time(self, time, velocities, f_0):
        if self.polynomial_coeffs is None:
            raise ValueError("Not enough history to fit a polynomial yet")            

        if time == self.history_t[-1]:
            dt = 0.0
        else:
            dt = 1.0
        grads = np.polyval(self.polynomial_coeffs, time-dt).reshape(self.history_grads[0].shape)
        prod = grads * velocities[:, None]
        axis = len(grads.shape) - 2
        delta_f = np.sum(prod, axis=axis)*dt
        extrap_f = f_0 + delta_f

        print(f'In Extrapolation: {time=} {dt=}')

        return extrap_f

    def extrapolate(self, t):
        if self.polynomial_coeffs is None:
            print(f'Not enough history to fit a polynomial yet for {self.name}: Returning Last Value')
            return self.history_grads[-1]
        print(f'Extrapolating {self.name} at time {t}; {self.history_t[-1]=}')
        values = np.polyval(self.polynomial_coeffs, t)
        return values.reshape(self.history_grads[0].shape)

class PolaritonLogger():
    name = 'polariton'
    def __init__(self) -> None:
        self._h5_file: H5File = None
        self._h5_group: h5py.Group
        self._initialized = False
        self._next_dataset: dict[str, np.ndarray] = {}
        self._labels: dict[str,list[str]] = {}

    def setup(self, logging_dir: str, h5_file: H5File):
        self._logging_Dir = logging_dir
        self._h5_file = h5_file
        self._h5_group = h5_file.create_group(self.name)
        self._h5_group.create_dataset('time', shape=(0,), maxshape=(None,), chunks=True)

    def _initialize(self):
        for key, data in self._next_dataset.items():
            ds = self._h5_group.create_dataset(key, shape=(0,)+data.shape, maxshape=(None,)+data.shape)
            if len(self._labels) > 0:
                ds.attrs.create('labels', self._labels[key])

    def set_labels(self, labels: dict[str,list[str]]):
        self._labels = labels.copy()

    def set_next_dataset(self, data: dict):
        self._next_dataset = data

    def write(self, logger_data: LoggerData):
        if not self._initialized:
            self._initialize()
        H5File.append_dataset(self._h5_group['time'], logger_data.time)
        for key, data in self._next_dataset.items():
            H5File.append_dataset(self._h5_group[key], data)

from openmm.openmm import CustomExternalForce

class TCPolaritonRunner(TCRunner):
        
    def __init__(self, coupled_mol: CoupledMolecule, atoms: list, tc_opts: TCRunnerOptions, max_wait=20, prev_ref_job: TCJob = None,) -> None:

        super().__init__(atoms, tc_opts, max_wait)

        self.coupled_mol = coupled_mol
        self.masses = np.array([[pt.to_mass(symbol)]*3 for symbol in atoms]).flatten()

        self._prev_evecs = None
        self._prev_ref_job = prev_ref_job

        self._polariton_logger = PolaritonLogger()
        self._tc_logger = TCJobsLogger()
        self._mol_sign_flipper = SignFlipper(len(coupled_mol.mol_grad_indices), 2, coupled_mol.n_nuclei*3, 'MOL')
        self._pol_sign_flipper = SignFlipper(coupled_mol.n_states, 2, coupled_mol.n_nuclei*3, 'POL')

        self._momentum_history = deque(maxlen=50)
        self._position_history = deque(maxlen=50)
        self._dipole_matrix_history = deque(maxlen=50)

        self._run_dipole_derivative_interpolation = False
        self._ran_actual_dipoles = False
        self._dpmd_tracker_gs = DipoleMatrixTracker(order=2, history_size=3, interval=10, name='GS')
        self._dpmd_tracker_ex = DipoleMatrixTracker(order=2, history_size=3, interval=10, name='EX')
        self._dpmd_tracker_tr = DipoleMatrixTracker(order=2, history_size=3, interval=10, name='TR')

        self._previous_pysces_outputs = None

        self._print_level = 1

        #   In the middle of an API change
        self._rk4_inteprolation = False
        self._interpolation = False

        #   load the previous state of the runner if it exists
        if os.path.isfile('_polariton_runner.pkl') and False:
            print('DEBUG: LOADING IN PREVIOUS POLARITON RUNNER STATE')
            with open('_polariton_runner.pkl', 'rb') as f:
                state = pickle.load(f)
                missing_in_pickle = self.__dict__.keys() - state.__dict__.keys()
                extra_in_pickle = state.__dict__.keys() - self.__dict__.keys()
                print(f'Missing objects in pickle: {missing_in_pickle}')
                print(f'Extra objects in pickle: {extra_in_pickle}')
                for key, val in list(state.__dict__.items()):

                    if key in ['_spec_job_opts', '_base_options', '_initial_frame_options']:
                        if val != self.__dict__[key]:
                            print('    TC Runner options have changed: ', key)
                            print('    New: ')
                            for k, v in self.__dict__[key].items():
                                print(f'        {k=}, {v=}')
                            print('    Old: ')
                            for k, v in val.items():
                                print(f'        {k=}, {v=}')
                            state.__dict__.pop(key)
                self.__setstate__(state.__dict__)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, str):
            if o == 'TCRunner' or o.lower() == 'terachem':
                return False
        return super().__eq__(o)

    def __getstate__(self):
        ''' Load in state for pickling.
            Any objects that are not picklable (mostly when they contian a socket object)
            are replaced with the NotPicklable class.
        '''
        state = self.__dict__.copy()
        for attr, value in list(state.items()):
            try:
                pickle.dumps(value)
            except pickle.PicklingError as e:
                print(f"{attr} is not picklable and will not be saved.")
                print(e)
                state.pop(attr)
            except TypeError as e:  # Some objects raise TypeError instead
                print(f"{attr} is not picklable and will not be saved.")
                print(e)
                state.pop(attr)
        return state
    
    def __setstate__(self, state):
        ''' Needed for pickling. 
            Any objects that are not picklable (mostly when they contian a socket object)
            should have been replaced with the NotPicklable class.
        '''
        for attr, value in state.items():
            self.__dict__[attr] = value
    
    def serialize(self):
        out_data = {}


    def save_state(self):
        pass
        # print('DEBUG: Saving Polariton Runner State')
        # with open('_polariton_runner.pkl', 'wb') as f:
        #     pickle.dump(self, f)

    def set_print_level(self, level):
        if level not in [0, 1, 2]:
            raise ValueError('Print level must be 0, 1, or 2')
        self._print_level = level

    @property
    def polariton_logger(self):
        return self._polariton_logger
    
    @property
    def tc_logger(self):
        return self._tc_logger
    
    @property
    def _n_steps(self):
        ''' Alias for the frame counter '''
        return self._frame_counter
    
    def state_data_to_matrix(self, gs_data, ex_data, tr_data):
        '''
            Converts ground state, excited state, and transition data into a matrix representation.

            Parameters
            -----------
            gs_data : numpy.ndarray
                Ground state data.
            ex_data : numpy.ndarray
                Excited state data.
            tr_data : numpy.ndarray
                Transition data.

            Returns
            --------
            numpy.ndarray
                A matrix representation of the state data with dimensions (n_elec, n_elec) + gs_data.shape,
                where n_elec is the number of electronic states.
        '''
       
        n_elec = self.coupled_mol._n_elec
        
        out_matrix = np.zeros((n_elec, n_elec,) + gs_data.shape)
        out_matrix[0, 0] = gs_data
        out_matrix[1:, 1:] = ex_data
        count = 0
        for i in range(n_elec):
            for j in range(i+1, n_elec):
                out_matrix[i, j] = tr_data[count]
                out_matrix[j, i] = tr_data[count]
                count += 1
        # out_matrix[np.triu_indices(n_elec, k=1)] = tr_data

        return out_matrix
    
    def matrix_to_state_data(self, matrix):
        n_elec = self.coupled_mol._n_elec
        gs_data = matrix[0, 0]
        ex_data = matrix[np.diag_indices(n_elec)][1:]
        tr_data = matrix[np.triu_indices(n_elec, k=1)]
        return gs_data, ex_data, tr_data

    def _send_jobs_to_clients(self, jobs_batch: TCJobBatch):
        ''' Overwrite the send jobs to clients method to add the dipole derivatives options '''
        
        if not self._interpolation:
            return super()._send_jobs_to_clients(jobs_batch)

        ''' need to fix the remaining'''

        mol = self.coupled_mol

        if not self._run_dipole_derivative_interpolation:
            super()._send_jobs_to_clients(jobs_batch)
            for j in jobs_batch.jobs:
                self._update_job_from_tcout(j)
            self.correct_signs(jobs_batch)
            
            mol.mol_dipole_matrix = self.dipole_matrix_from_job(jobs_batch.jobs[-1])
            mol.mol_dipole_matrix_gradient = self.get_all_dipole_gradients_from_jobs(jobs_batch)
            self._dipole_matrix_history.append((self._n_steps, mol.mol_dipole_matrix))
            return jobs_batch

        gs_job, ex_job, tr_job = None, None, None
        for job in jobs_batch.jobs:
            if 'dipolederivative' in job.opts:
                gs_job: TCJob = job
            if 'cisdipolederiv' in job.opts:
                ex_job: TCJob = job
            if 'cistransdipolederiv' in job.opts:
                tr_job: TCJob = job

        gs_run, gs_reason = self._dpmd_tracker_gs.check_run_deriv(self._n_steps)
        ex_run, ex_reason = self._dpmd_tracker_ex.check_run_deriv(self._n_steps)
        tr_run, tr_reason = self._dpmd_tracker_tr.check_run_deriv(self._n_steps)

        gs_job.opts['dipolederivative'] = 'yes' if gs_run else 'no'
        ex_job.opts['cisdipolederiv'] = 'yes' if ex_run else 'no'
        tr_job.opts['cistransdipolederiv'] = 'yes' if tr_run else 'no'

        if gs_run:
            print('Computing GS Dipole Derivative: ', gs_reason)
        if ex_run:
            print('Computing EX Dipole Derivative: ', ex_reason)
        if tr_run:
            print('Computing TR Dipole Derivative: ', tr_reason)


        super()._send_jobs_to_clients(jobs_batch)
        ref_job = self.correct_signs(jobs_batch)

        for job in jobs_batch.jobs:
            if job.name == 'gradient_1':
                mol.mol_dipole_matrix = self.dipole_matrix_from_job(job)
                self._dipole_matrix_history.append((self._n_steps, mol.mol_dipole_matrix))
                break

        dipoles_gs, dipoles_ex, dipoles_tr = self.matrix_to_state_data(mol.mol_dipole_matrix)

        #   update histories
        updated_gs, updated_ex, updated_tr = False, False, False
        if gs_job.opts['dipolederivative'] == 'yes':
            updated_gs = True
            gs_dipole_grad = self.get_gs_dipole_gradient_from_jobs(jobs_batch)
            self._dpmd_tracker_gs.update_history(self._n_steps, gs_dipole_grad, dipoles_gs)
        if ex_job.opts['cisdipolederiv'] == 'yes':
            updated_ex = True
            ex_dipole_grad = self.get_ex_dipole_gradient_from_jobs(jobs_batch)
            self._dpmd_tracker_ex.update_history(self._n_steps, ex_dipole_grad, dipoles_ex)
        if tr_job.opts['cistransdipolederiv'] == 'yes':
            updated_tr = True
            tr_dipole_grad = self.get_tr_dipole_gradient_from_jobs(jobs_batch)
            self._dpmd_tracker_tr.update_history(self._n_steps, tr_dipole_grad, dipoles_tr)
        
        if updated_gs and updated_ex and updated_tr:
            mol.mol_dipole_matrix_gradient = self.state_data_to_matrix(gs_dipole_grad, ex_dipole_grad, tr_dipole_grad)
            print('All dipole derivatives updated')
            return jobs_batch

        gs_redo, ex_redo, tr_redo = self.check_dipole_matrix_accuracy()
        new_job_batch = TCJobBatch()

        if gs_redo:
            print('    GS REDO AT STEP ', self._n_steps)
            new_gs_job = gs_job.new_from_old()
            new_gs_job.opts['dipolederivative'] = 'yes'
            new_gs_job.job_type = 'energy'
            new_job_batch.append(new_gs_job, allow_duplicates=False)
        if ex_redo:
            print('    EX REDO AT STEP ', self._n_steps)
            new_ex_job = ex_job.new_from_old()
            new_ex_job.opts['cisdipolederiv'] = 'yes'
            new_ex_job.job_type = 'energy'
            new_job_batch.append(new_ex_job, allow_duplicates=False)
        if tr_redo:
            print('    TR REDO AT STEP ', self._n_steps)
            new_tr_job = tr_job.new_from_old()
            new_tr_job.opts['cistransdipolederiv'] = 'yes'
            new_tr_job.job_type = 'energy'
            new_job_batch.append(new_tr_job, allow_duplicates=False)

        if len(new_job_batch) > 0:
            super()._send_jobs_to_clients(new_job_batch)
            self.correct_signs(new_job_batch, ref_job)
        
            if gs_redo:
                gs_dipole_grad = self.get_gs_dipole_gradient_from_jobs(new_job_batch)
                self._dpmd_tracker_gs.update_history(self._n_steps, gs_dipole_grad, dipoles_gs)
            if ex_redo:
                ex_dipole_grad = self.get_ex_dipole_gradient_from_jobs(new_job_batch)
                self._dpmd_tracker_ex.update_history(self._n_steps, ex_dipole_grad, dipoles_ex)
            if tr_redo:
                tr_dipole_grad = self.get_tr_dipole_gradient_from_jobs(new_job_batch)
                self._dpmd_tracker_tr.update_history(self._n_steps, tr_dipole_grad, dipoles_tr)

        gs_dipole_grad = self._dpmd_tracker_gs.extrapolate(self._n_steps)
        ex_dipole_grad = self._dpmd_tracker_ex.extrapolate(self._n_steps)
        tr_dipole_grad = self._dpmd_tracker_tr.extrapolate(self._n_steps)
        mol.mol_dipole_matrix_gradient = self.state_data_to_matrix(gs_dipole_grad, ex_dipole_grad, tr_dipole_grad)


        # if self._n_steps % 1 == 0 and self._n_steps > 0:
        #     input('Continue?')
        return jobs_batch

    def check_dipole_matrix_accuracy(self) -> tuple[bool, bool, bool]:

        ready = self._dpmd_tracker_gs.check_if_ready()
        ready *= self._dpmd_tracker_ex.check_if_ready()
        ready *= self._dpmd_tracker_tr.check_if_ready()
        if not ready:
            return False, False, False

        print('Previous Momentum time '     , self._momentum_history[-2][0], 'Current time: ', self._n_steps)
        print('Previous Dipole-matrix time ', self._dipole_matrix_history[-2][0], 'Current time: ', self._n_steps)


        velocities = self._momentum_history[-2][1]/(self.masses * AMU_2_AU)
        mol_dipole_matrix = self._dipole_matrix_history[-2][1]
        gs_dipoles, ex_dipoles, tr_dipoles = self.matrix_to_state_data(mol_dipole_matrix)
        gs_dipoles = self._dpmd_tracker_gs.predict_at_time(self._n_steps, velocities, gs_dipoles)
        ex_dipoles = self._dpmd_tracker_ex.predict_at_time(self._n_steps, velocities, ex_dipoles)
        tr_dipoles = self._dpmd_tracker_tr.predict_at_time(self._n_steps, velocities, tr_dipoles)
        extrap_dipole_mat = self.state_data_to_matrix(gs_dipoles, ex_dipoles, tr_dipoles)

        actual_dipole_mat = self._dipole_matrix_history[-1][1]
        error_matrix = np.linalg.norm(extrap_dipole_mat - actual_dipole_mat, axis=2)

        #   generate stats of dipole differences
        actual_mags = np.linalg.norm(actual_dipole_mat, axis=2)
        extrap_mags = np.linalg.norm(extrap_dipole_mat, axis=2)
        diff_mags = np.abs(extrap_mags - actual_mags)
        pct_diff_mags = 100*diff_mags/actual_mags

        gs_redo, ex_redo, tr_redo = False, False, False
        cutoff = 0.005
        print()
        print('               Mag. Diff.   % Diff.   Actual Mag.  Extrap Mag.   Error')
        print('--------------------------------------------------------------------------')
        print('\nGround state:')
        print('--------------------')
        print(f'DIPDIF  0   0   {diff_mags[0, 0]:10.6f}  {pct_diff_mags[0, 0]:10.6f}  {actual_mags[0, 0]:10.6f}  {extrap_mags[0, 0]:10.6f}  {error_matrix[0, 0]:10.6f} ', '*'*(gs_redo))
        print('\nExcited states:')
        print('--------------------')
        for i in range(1, mol_dipole_matrix.shape[0]):
            if error_matrix[i, i] > cutoff:
                ex_redo = True
            flag_str = '*'*(error_matrix[i, i] > cutoff)
            print(f'DIPDIF  {i}   {i}   {diff_mags[i, i]:10.6f}  {pct_diff_mags[i, i]:10.6f}  {actual_mags[i, i]:10.6f}  {extrap_mags[i, i]:10.6f}  {error_matrix[i, i]:10.6f} ', flag_str)
        print('\nGs-Ex Transitions:')
        print('--------------------')
        for j in range(1, mol_dipole_matrix.shape[0]):
            if error_matrix[0, j] > cutoff:
                tr_redo = True
            flag_str = '*'*(error_matrix[0, j] > cutoff)
            print(f'DIPDIF  {0}   {j}   {diff_mags[0, j]:10.6f}  {pct_diff_mags[0, j]:10.6f}  {actual_mags[0, j]:10.6f}  {extrap_mags[0, j]:10.6f}  {error_matrix[0, j]:10.6f}', flag_str)
        print('\nEx-Ex Transitions:')
        print('--------------------')
        for i in range(1, mol_dipole_matrix.shape[0]):
            for j in range(i+1, mol_dipole_matrix.shape[1]):
                #   We don't care about transitions between two excited states
                error_matrix[i, j] = 0.0
                print(f'DIPDIF  {i}   {j}   {diff_mags[i, j]:10.6f}  {pct_diff_mags[i, j]:10.6f}  {actual_mags[i, j]:10.6f}  {extrap_mags[i, j]:10.6f}  {error_matrix[i, j]:10.6f}')

        #   check if we need to re any of the caluclations
        gs_redo = bool(error_matrix[0, 0] > cutoff)

        return (gs_redo, ex_redo, tr_redo)

    def correct_signs(self, job_batch: TCJobBatch, ref_job=None):
        if ref_job is None:
            # use the highest gradient state as the reference job
            grad_batch = job_batch.get_by_type('gradient')
            curr_ref_job = grad_batch.sorted_jobs_by_state()[-1]
        else:
            curr_ref_job = ref_job

        if self._prev_ref_job is None:
            self._update_job_from_tcout(curr_ref_job)
            self._prev_ref_job = curr_ref_job

        #   update jobs form tc.out file, and correct with esp charges
        for job in job_batch.jobs:
            self._update_job_from_tcout(job)
            #   ground state jobs don't have transition dipoles
            if job.state == 0:
                continue

            TeraChem._correct_signs(job, self._prev_ref_job)

        self._prev_ref_job = curr_ref_job
        return curr_ref_job

    def run_new_geom(self, phase_vars: 'PhaseVars' = None, geom=None, momentum=None):

        if phase_vars is not None:
            geom = phase_vars.nuc_q*BOHR_2_ANG
        elif geom is not None:
            #   legacy support for geom, assumed to be in angstroms
            pass
        else:
            raise ValueError('Either phase_vars or geom must be provided')

        
        self._position_history.append((self._n_steps, geom))
        self._momentum_history.append((self._n_steps, momentum))

        #   step 1
        dipoles = np.arange(0, max(self._grads) + 1, dtype=int).tolist()
        tr_dipoles = [(0, x) for x in range(1, max(self._grads) + 1)] + self._NACs
        job_batch = self.create_jobs(geom, False, self._grads, self._NACs, dipoles, tr_dipoles)
        input('Continue?')
        job_batch = self._send_jobs_to_clients(job_batch)

        #   step 2
        if self._interpolate_grads or self._interpolate_NACs:
            new_grads, new_nacs = self._check_new_grads_nacs_to_run(job_batch)
            new_dipole_grads, new_tr_dipole_grads = self._check_new_dipole_grads_to_run(job_batch)
            job_batch_2 = self.create_jobs(geom, False, new_nacs, new_grads, new_dipole_grads, new_tr_dipole_grads)
            job_batch_2 = self._send_jobs_to_clients(job_batch_2)

            #   step 3: combine both batches
            job_batch.jobs += job_batch_2.jobs

        self._log_jobs(job_batch, self._frame_counter)
        self.compute_coupled_mol_properties(job_batch.results_list)
        self.log_timestep()
        self.print_results()
        self.set_pysces_outputs()
        self.save_state()
        self._finalize_frame(job_batch)

        return self.get_pysces_outputs()
        
    def _check_new_dipole_grads_to_run(self, job_batch: TCJobBatch):
        if not self._run_dipole_derivative_interpolation:
            return (), ()
        else:
            raise NotImplementedError('Interpolation of dipole gradients not yet implemented')

    def _correct_nac_sign_flips(self, nacs, trans_dips):
        sub_nacs = nacs[np.ix_(self._grads, self._grads)]
        sub_trans_dips = trans_dips[np.ix_(self._grads, self._grads)]
        if self._n_steps == 0:
            self._mol_sign_flipper.set_history(sub_nacs, np.empty(0), sub_trans_dips, np.empty(0))
        sub_nacs = self._mol_sign_flipper.correct_nac_sign(sub_nacs, sub_trans_dips)

        self._initialize_nac_sign(nacs)

    def compute_coupled_mol_properties(self, results_list: list[dict]):
        all_states = np.arange(0, max(self._grads) + 1)
        all_energies, elecE, grads, nacs, dipole_matrix, dipole_matrix_grads = format_combo_job_results(results_list, all_states)

        #   correct for sign flips
        self._correct_nac_sign_flips(nacs, dipole_matrix)

        #   update the dipole matrixcompute all polariton properties
        self.coupled_mol.compute_all(all_energies, grads, nacs, dipole_matrix, dipole_matrix_grads, self._prev_evecs)

        self._prev_evecs = self.coupled_mol.eigen_vecs

    def set_pysces_outputs(self):
        #   TODO: Compute transition dipoles!!!
        mol = self.coupled_mol
        if 0 not in self.coupled_mol.mol_grad_indices:
            out_eigen_vals = mol.eigen_vals[1:]
            out_eigen_val_grads = mol.eigen_val_gradients[1:]
            out_NACs = mol.NACs[1:, 1:]
        else:
            out_eigen_vals = mol.eigen_vals
            out_eigen_val_grads = mol.eigen_val_gradients
            out_NACs = mol.NACs


        if self._rk4_inteprolation:
            self._previous_pysces_outputs = None
            raise NotImplementedError('RK4 interpolation not yet implemented')
        else:
            self._previous_pysces_outputs = (mol.eigen_vals, out_eigen_vals, out_eigen_val_grads, out_NACs, None, None)

    def get_pysces_outputs(self):
        return self._previous_pysces_outputs

    def log_timestep(self):
        #   log all computed quantities
        mol = self.coupled_mol
        logged_data = {}
        logged_data['hamiltonian'] = mol.hamiltonian
        logged_data['eigenvalues'] = mol.eigen_vals
        logged_data['eigenvectors'] = mol.eigen_vecs
        logged_data['hamiltonian_grads'] = mol.dH
        logged_data['eigenvalue_grads'] = mol.eigen_val_gradients
        logged_data['NACs'] = mol.NACs
        logged_data['dipole_matrix'] = mol.mol_dipole_matrix
        logged_data['dipole_matrix_grads'] = mol.mol_dipole_matrix_gradient
        self.polariton_logger.set_next_dataset(logged_data)

    def print_results(self):
        if self._print_level == 0:
            return
        mol = self.coupled_mol

        print(' ########## Polariton Addon ##########')
        fld = mol._field_dir
        fld_mag = np.linalg.norm(fld)
        print(f'Field Direction: [{fld[0]:.3f}, {fld[1]:.3f}, {fld[2]:.3f}]\n')
        print('State dipole moments and angle with cavity field:\n')
        print('   Root         Dx         Dy         Dz        |D|      Theta   (a.u./Degrees)')
        print('-----------------------------------------------------------------------------------')
        for i in range(mol.mol_dipole_matrix.shape[0]):
            mu = mol.mol_dipole_matrix[i, i]
            angle = np.arccos(np.dot(mu, fld)/(np.linalg.norm(mu)*fld_mag)) * 180/np.pi
            print('    {:2d}  {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.3f}'.format(i, *mu, np.linalg.norm(mu), angle))
        print('\n')

        print('Transition dipoles moments and angle with cavity field:\n')
        print('    Transition         Dx         Dy         Dz        |D|      Theta   (a.u./Degrees)')
        print('-----------------------------------------------------------------------------------------')
        for i in range(0, mol.mol_dipole_matrix.shape[0]):
            for j in range(i+1, mol.mol_dipole_matrix.shape[0]):
                mu = mol.mol_dipole_matrix[i, j]
                angle = np.arccos(np.dot(mu, fld)/(np.linalg.norm(mu)*fld_mag)) * 180/np.pi
                print('    {:2d} ->  {:2d}  {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.3f}'.format(i, j, *mu, np.linalg.norm(mu), angle))
        print('\n')

        print('Polariton Hamiltonian Diagonal Elements (eV):\n')
        print('  State   (alpha, n)   Energy (a.u.)  Ex Energy     H_en      H_p      H_d')
        print('----------------------------------------------------------------------------')
        min_ham_energy = np.min(mol.hamiltonian)
        min_en_energy = np.min(mol.mol_energies)
        for i in range(mol.hamiltonian.shape[0]):
            ex_energy = (mol.hamiltonian[i, i] - min_ham_energy)*AU_2_EV
            state = mol.state_pairs[i]
            H_en = (mol._H_en[i, i] - min_en_energy)*AU_2_EV
            H_p, H_d = mol._H_p[i, i]*AU_2_EV, mol._H_d[i, i]*AU_2_EV
            print(f'   {i:2d}      ({state[0]:2d}, {state[1]:2d})  {mol.hamiltonian[i, i]:12.8f}   {ex_energy:10.6f}  {H_en:7.4f}  {H_p:7.4f}  {H_d:7.4f}')
        print('\n')

        print('Largest off diagonal elements of the Hamiltonian:\n')
        print('  Basis(i, j)  (alpha,m)  (beta,n)   Energy (a.u.)    Energy (eV)')
        print('---------------------------------------------------------------------------------------------')
        off_diags = np.abs(mol.hamiltonian - np.diag(np.diag(mol.hamiltonian)))
        sorted_indices = np.argsort(off_diags, axis=None)
        sorted_2d_indices = np.unravel_index(sorted_indices, off_diags.shape)
        sorted_2d_indices = np.column_stack(sorted_2d_indices)

        count = 0
        used_pairs = []
        for i, j in reversed(sorted_2d_indices):
            state_i = mol.state_pairs[i]
            state_j = mol.state_pairs[j]
            if ((state_j, state_i) in used_pairs):
                continue
            
            ham_value = mol.hamiltonian[i, j]
            # H_en    = mol._H_en[i, j]*AU_2_EV
            # H_p     = mol._H_p[i, j]*AU_2_EV
            # H_d     = mol._H_d[i, j]*AU_2_EV
            # H_en_p  = mol._H_en_p[i, j]*AU_2_EV

            print(f'     {(int(i), int(j))}       {state_i}    {state_j}  {ham_value:12.8f}    {(ham_value)*AU_2_EV:10.6f}')

            count += 1
            used_pairs.append((state_i, state_j))
            if count >= 10:
                break
        print('\n')

        print('Wavefunctions:')
        for i in range(mol.eigen_vecs.shape[1]):
            eig_val = (mol.eigen_vals[i] - min_ham_energy)*AU_2_EV
            print(f'State {i}  ({eig_val:4.2f}eV): Largest coefficients/occupations:')
            e_vec = mol.eigen_vecs[:, i]
            idx = np.argsort(np.abs(e_vec))
            for j in reversed(idx[-5:]):
                state = mol.state_pairs[j]
                print(f'    {j}: {state}  {e_vec[j]:7.4f} {e_vec[j]**2:7.4f}')
            print('\n')
        print('\n')
        

        # with open('coupled_mol.pkl', 'wb') as file:
        #     pickle.dump(mol, file)

        if True:
            self.print_dipole_derivatives()
    
    def print_dipole_derivatives(self):
        if self._print_level < 1:
            return
        mol = self.coupled_mol
        print('Molecular Dipole Moment Derivatives (a.u.):\n')
        for i in range(mol.mol_dipole_matrix_gradient.shape[0]):
            for j in range(i, mol.mol_dipole_matrix_gradient.shape[1]):
                print(' State Pair: ', i, j)
                print('   Atom        dMuX       dMuY       dMuZ         dMu*Mu   dMu*Field')
                print('-----------------------------------------------------------------------')
                for k in range(mol.mol_dipole_matrix_gradient.shape[2]):
                    grad = mol.mol_dipole_matrix_gradient[i, j, k]
                    dipole = mol.mol_dipole_matrix[i, j]
                    in_field_direction = np.dot(grad, mol._field_dir)
                    in_dipole_direction = np.dot(grad, dipole)/np.linalg.norm(dipole)
                    print(f'    {k+1:2d}   {grad[0]:10.5f} {grad[1]:10.5f} {grad[2]:10.5f}     {in_dipole_direction:10.5f}  {in_field_direction:10.5f}')
                print('---')
                print('\n')
        print('\n')



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
    
        job_data = parse_from_list(job.results['tc.out']).model_dump(mode='json')

        for key in job_data:
            if key not in job.results:
                job.results[key] = job_data[key]

    # def dipole_matrix_from_job(self, tc_job: TCJob):
    def dipole_matrix_from_job(self, tc_job_results: dict):
        '''
            re-order and combine all of the the dipoles from a TC job dict
            into a single matrix.
        '''

        excited_type = self._excited_type
        if excited_type == 'cis':
            dipole_key = 'cis_unrelaxed_dipoles'
            tr_dipole_key = 'cis_transition_dipoles'
        elif excited_type == 'cas':
            dipole_key = 'cas_dipoles'
            tr_dipole_key = 'cas_transition_dipoles'
        else:
            raise ValueError(f"Invalid job type specified '{excited_type}'; must be 'cis' or 'cas'")

        # job_results = tc_job.results
        results = tc_job_results
        n_states = len(results['energy'])
        dipole_matrix = np.zeros((n_states, n_states, 3))
        dipole_matrix[0, 0] = np.array(results['dipole_vector'])*DEBYE_2_AU
        for i in range(1, n_states):
            dipole_matrix[i, i] = results[dipole_key][i-1]
        indicies = np.transpose(np.triu_indices(n_states, k=+1))
        for count, (i, j) in enumerate(indicies):
            dipole_matrix[i, j] = results[tr_dipole_key][count]
            dipole_matrix[j, i] = results[tr_dipole_key][count]

        return dipole_matrix
    
    def get_gs_dipole_gradient_from_jobs(self, jobs_batch: TCJobBatch):
        '''
            extract ground state dipole matrix from a TeraChem job batch

            Parameters
            ----------
            jobs_batch: TCJobBatch
                batch of jobs to extract the dipole derivatives from
            
            Returns
            -------
            np.ndarray: dipole_grads (n_states, 3 * n_atoms, 3)
                dipole_grads[j, k] is the dipole gradient with respect to the jth atom and kth cartesian coordinate
        '''
        dipole_grads = np.zeros_like(self.coupled_mol.mol_dipole_matrix_gradient[0, 0])
        got_gs = False
        for tc_job in jobs_batch.jobs:
            if 'dipole_deriv' in tc_job.results:
                derivs = np.array(tc_job.results['dipole_deriv'])
                dipole_grads = derivs.transpose((1, 2, 0)).reshape(-1, 3)
                got_gs = True

        if not got_gs:
            raise ValueError('Could not recover ground state dipole moment derivatives from TC jobs')
 
        return dipole_grads   
    
    def get_ex_dipole_gradient_from_jobs(self, jobs_batch: TCJobBatch):
        '''
            extract excited state dipole matrix from a TeraChem job batch

            Parameters
            ----------
            jobs_batch: TCJobBatch
                batch of jobs to extract the dipole derivatives from
            
            Returns
            -------
            np.ndarray: dipole_grads (n_states, 3 * n_atoms, 3)
                dipole_grads[i, j, k] is the dipole gradient for the ith excited state
                with respect to the jth atom and kth cartesian coordinate
        '''
        n_ex_states = self.coupled_mol._n_elec - 1
        dipole_grads = np.zeros((n_ex_states, self.coupled_mol.n_nuclei*3, 3))
        got_ex = False
        # search_key = 'cis_unrelaxed_dipole_deriv'
        search_key = 'cis_dipole_deriv'


        # for j in jobs_batch.jobs:
        #     with open(f'job_{j.name}.txt', 'w') as file:
        #         pprint(j.results, file)


        for tc_job in jobs_batch.jobs:
            if search_key in tc_job.results:
                derivs = np.array(tc_job.results[search_key])
                derivs = derivs.transpose((0, 2, 3, 1)).reshape(n_ex_states, -1, 3)
                for i in range(0, n_ex_states):
                    dipole_grads[i] = derivs[i]
                got_ex = True

        if not got_ex:
            raise ValueError('Could not recover excited state dipole moment derivatives from TC jobs')
 
        return dipole_grads
    
    def get_tr_dipole_gradient_from_jobs(self, jobs_batch: TCJobBatch):
        '''
            extract transition state dipole matrix from a TeraChem job batch

            Parameters
            ----------
            jobs_batch: TCJobBatch
                batch of jobs to extract the dipole derivatives from

            Returns
            -------
            np.ndarray: dipole_grads (n_states, n_states, 3 * n_atoms, 3)
                dipole_grads[i, j, k] is the dipole gradient for the ith transition
                with respect to the jth atom and kth cartesian coordinate. The i-th 
                Transitions are ordered as (0, 1), (0, 2), ..., (0, n), (1, 2), ..., (n-1, n)
        '''
        dipole_grads = np.zeros_like(self.coupled_mol.mol_dipole_matrix_gradient)
        n_ex_states = self.coupled_mol._n_elec - 1
        n_grads = int((n_ex_states*(n_ex_states+1))/2)
        dipole_grads = np.zeros((n_grads, self.coupled_mol.n_nuclei*3, 3))

        got_tr = False
        for tc_job in jobs_batch.jobs:
            if 'cis_transition_dipole_deriv' in tc_job.results:
                derivs = np.array(tc_job.results['cis_transition_dipole_deriv'])
                #   swap second and 4th axis. The last axis is now mX,mY,mZ
                #   then, flatten the middle two axis, which are the cartesian coordinates
                dipole_grads = derivs.transpose((0, 2, 3, 1)).reshape(n_grads, -1, 3)
                # indicies = np.transpose(np.triu_indices(n_ex_states+1, k=+1))
                # for count, (i, j) in enumerate(indicies):
                #     dipole_grads[i, j] = derivs[count]
                #     dipole_grads[j, i] = derivs[count]
                got_tr = True

        if not got_tr:
            raise ValueError('Could not recover transition state dipole moment derivatives from TC jobs')
 
        return dipole_grads
    
    def get_all_dipole_gradients_from_jobs(self, jobs_batch: TCJobBatch):
        """
        Retrieve and process all dipole gradients from a batch of jobs.

        This method extracts ground state (GS), excited state (EX), and transition (TR) dipole gradients
        from the provided job batch, prints their shapes and contents for debugging purposes, and then
        converts the state data into a matrix format.

        Args:
            jobs_batch (TCJobBatch): A batch of jobs containing the necessary data to extract dipole gradients.

        Returns:
            np.ndarray: A matrix containing the processed dipole gradients for all states of shape (n_states, n_states, n_atoms, 3).
        """
        gs_grads = self.get_gs_dipole_gradient_from_jobs(jobs_batch)
        ex_grads = self.get_ex_dipole_gradient_from_jobs(jobs_batch)
        tr_grads = self.get_tr_dipole_gradient_from_jobs(jobs_batch)


        return self.state_data_to_matrix(gs_grads, ex_grads, tr_grads)

            

    def dipole_matrix_gradient_from_jobs_OLD(self, jobs_batch: TCJobBatch, partial_ok=False):
        '''
            re-order and combine all of the the dipole derivatives from a list of 
            TC jobs into a single matrix.
        '''
        dipole_grads = np.zeros_like(self.coupled_mol.mol_dipole_matrix_gradient)
        # n_ex_states = self.coupled_mol._s_high - min(self.coupled_mol._s_low, 1) + 1
        n_ex_states = self.coupled_mol._n_elec - 1

        got_gs, got_ex, got_tr = False, False, False
        for tc_job in jobs_batch.jobs:
            if 'cis_transition_dipole_deriv' in tc_job.results:
                derivs = np.array(tc_job.results['cis_transition_dipole_deriv'])
                #   swap second and 4th axis. The last axis is now mX,mY,mZ
                #   then, flatten the middle two axis, which are the cartesian coordinates
                n_elms = n_ex_states*(n_ex_states+1)//2
                derivs = derivs.transpose((0, 2, 3, 1)).reshape(n_elms, -1, 3)

                indicies = np.transpose(np.triu_indices(n_ex_states+1, k=+1))
                for count, (i, j) in enumerate(indicies):
                    dipole_grads[i, j] = derivs[count]
                    dipole_grads[j, i] = derivs[count]
                got_tr = True

            if 'cis_dipole_deriv' in tc_job.results:

                derivs = np.array(tc_job.results['cis_dipole_deriv'])
                derivs = derivs.transpose((0, 2, 3, 1)).reshape(n_ex_states, -1, 3)
                for i in range(1, n_ex_states+1):
                    dipole_grads[i, i] = derivs[i-1]
                got_ex = True

            if 'dipole_deriv' in tc_job.results:
                derivs = np.array(tc_job.results['dipole_deriv'])
                derivs = derivs.transpose((1, 2, 0)).reshape(-1, 3)
                dipole_grads[0, 0] = derivs
                got_gs = True

        if not got_gs or not got_ex or not got_tr:
            print('TC JOBS:')
            from pprint import pprint
            for job in tc_jobs:
                print(job.name)
                for k, v in job.results.items():
                    print(f'{k}:')
                    pprint(v)
                print(json.dumps(job.results, indent=4))
                print('\n\n')

        if not got_gs and not partial_ok:
            raise ValueError('Could not recover ground state dipole moment derivatives from TC jobs')
        if not got_ex and not partial_ok:
            raise ValueError('Could not recover excited state dipole moment derivatives from TC jobs')
        if not got_tr and not partial_ok:
            raise ValueError('Could not recover transition state dipole moment derivatives from TC jobs')

        return dipole_grads

