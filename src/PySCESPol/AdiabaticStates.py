
import numpy as np
from . import NumDeriv as numD

class AdiabaticStates():
    def __init__(self, n_states, n_nuclei, subset_indices: list[int, int] | None = None) -> None:
        # N = n_states
        N = len(subset_indices) if subset_indices is not None else n_states

        self._n_states = N
        self._n_nuclei = n_nuclei
        self._hamiltonian = np.zeros((N, N))
        self._dH = np.zeros((N, N, n_nuclei*3))
        self._diagonalized = False

        self.eigen_vals = np.zeros(N)
        self.eigen_vecs = np.zeros((N, N))
        self.NACs = np.zeros((N, N, n_nuclei*3))
        self.eigen_val_gradients = np.zeros((N, n_nuclei*3))
        self.eigen_vec_gradients = np.zeros((N, N, n_nuclei*3))
        

    @property
    def n_states(self): return self._n_states

    @property
    def n_nuclei(self): return self._n_nuclei

    @property
    def hamiltonian(self):
        return self._hamiltonian
        
    @hamiltonian.setter
    def hamiltonian(self, matrix: np.ndarray):
        # self.zero()
        if matrix.shape != self._hamiltonian.shape:
            raise ValueError(f'Atempting to set Hamiltonian to a size of {matrix.shape} when it should be {self._hamiltonian.shape}')
        self._hamiltonian = matrix
        self._hamiltonian.flags['WRITEABLE'] = False

    @property
    def dH(self):
        return self._dH
    
    @dH.setter
    def dH(self, matrix: np.ndarray):
        if matrix.shape != self._dH.shape:
            raise ValueError(f'Atempting to set Hamiltonian gradient to a size of {matrix.shape} when it should be {self._dH.shape}')
        self._dH = matrix
        self._dH.flags['WRITEABLE'] = False

    def zero(self):
        '''
            Set all properties of the AdiabaticStates object to zero.
        '''
        self._hamiltonian  = np.zeros_like(self._hamiltonian)
        self._dH = np.zeros_like(self._dH)
        self._diagonalized = False

        for x in (self.eigen_vals, self.dH, self.NACs, self.eigen_val_gradients, self.eigen_vec_gradients):
            x = np.zeros(x.shape)

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
            Compute the gradients of the eigenvectors with respect to the nuclear coordinates.

            Returns
            -------
            e_vec_grads: np.ndarray (n_states, n_states, 3*n_nuclei)
        '''

        e_vec_grads = np.zeros((self._n_states, self._n_states, self._n_nuclei*3))
        for i, E_i in enumerate(self.eigen_vals):
            C_i = self.eigen_vecs[:, i]
            for j, E_j in enumerate(self.eigen_vals):
                C_j = self.eigen_vecs[:, j]
                if i == j: continue
                inverse_energies = 1/(E_i - E_j)
                C_dH_C = np.einsum('i,ijk,j->k', C_j, self.dH, C_i)
                e_vec_grads[:, i] += inverse_energies * C_dH_C * C_j[:, None]

        self.eigen_vec_gradients = e_vec_grads
        self.eigen_vec_gradients.flags['WRITEABLE'] = False
        return self.eigen_vec_gradients

    def _calc_eigen_vector_gradient_reference(self):
        '''
            This function is teribly slow. Only kept for reference.
            Use calc_eigen_vector_gradient() instead.
        '''
        e_vec_grads = np.zeros((self._n_states, self._n_states, self._n_nuclei*3))
        for s in range(self._n_states):
            for i, E_i in enumerate(self.eigen_vals):
                C_i = self.eigen_vecs[:, i]
                for j, E_j in enumerate(self.eigen_vals):
                    C_j = self.eigen_vecs[:, j]
                    if i == j: continue
                    inverse_energies = 1/(E_i - E_j)
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
