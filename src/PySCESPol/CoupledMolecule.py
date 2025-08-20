
import numpy as np
from copy import deepcopy
from .AdiabaticStates import AdiabaticStates
from math import sqrt

class CoupledMolecule(AdiabaticStates):

    def __init__(self, 
                 omega_c: float, 
                 mol_grads: list[int], 
                 n_nuc: int, 
                 field_dir: list[float] | None = None, 
                 subset_states: str | list[list[int, int]] | None = None,
                 rwa: bool = False,
                 dse: bool = True,
                 pdt: bool = True,
                 ):
        '''
            Create a molecule that is coupled to a polariton cavity

            Parameters
            ----------
            omega_c : float
                The cavity frequency in atomic units.
            mol_grads : list[int]
                Indices of molecular gradients to be included in the calculation.
            n_nuc : int
                The number of nuclei in the molecule.
            field_dir : list[float] | None, optional
                The direction of the electric field as a 3D vector. If None, the field direction is assumed to be aligned with each of transition dipoles for each Hamiltonian matrix element (Debug only).
            subset_states : str | list[list[int, int]] | None, optional
                Specifies a subset of states to include in the calculation. 
                - If "all" or None, all states are included
                - If "single", only the first excitation manifold of states is used. That is, the photon dressed molecular ground state + the first excited state of each molecule.
                - Otherwise, a list of pairs of integers (a, n) where a is the index of the molecular state and n is the index of the polariton state.
            rwa : bool, optional
                Whether to use the Rotating Wave Approximation (RWA), which drops the counter-rotating terms as well as the permanent molecular dipoles. Default is False.
            dse : bool, optional
                Whether to include the Dipole Self-Energy (DSE) term. Default is True.
        '''

        self.mol_grad_indices = mol_grads
        n_elec = max(mol_grads) + 1
        # n_elec = len(mol_grads)

        # super().__init__(n_elec*2, n_nuc)
        self._field_dir = field_dir
        if self._field_dir is not None:
            self._field_dir = field_dir/np.linalg.norm(field_dir)
        self._omega_c = omega_c
        self._gc = None
        self._n_elec = n_elec

        n_pol = 2
        self._state_pairs = np.zeros((n_pol * self._n_elec, 2), dtype=int)
        for n in range(n_pol):
            for a in range(n_elec):
                count = n*n_elec + a
                self._state_pairs[count] = (a, n)
        self._state_pairs = tuple(tuple(p.tolist()) for p in self._state_pairs)
        self._n_dim = len(self._state_pairs)

        self._subset_state_indices = self._calc_subset_indices(subset_states)
        #   override the state pairs to only include the subset of states
        self._state_pairs = tuple(self._state_pairs[i] for i in self._subset_state_indices)
        self._n_dim = len(self._state_pairs) 
        super().__init__(self._n_dim, n_nuc)

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

        #   turns on/off Hamiltonian approximations
        self._use_DSE = dse # dipole self energy
        self._use_RWA = rwa # use the rotating wave approximation
        self._use_PDT = pdt # use the permanent dipoles term in the Hamiltonian



    def _calc_subset_indices(self, subset_states: str | list[list[int, int]] | None):
        if subset_states == 'single':
            #   only the first excitation manifold of states is used
            #   that is, the photon dressed molecular ground state + the first excited state of each molecule
            subset_states = [(a, 0) for a in range(1, self._n_elec)] + [(0, 1)]

            indices = []
            for i, pair in enumerate(self.state_pairs):
                if pair in subset_states:
                    indices.append(i)

            if len(indices) == 0:
                raise ValueError(f'No states found in the subset {subset_states} for the molecule with {self._n_elec} electronic states.')

        elif subset_states == 'all' or subset_states is None:
            indices = list(range(self._n_dim))


        #   print the states we are using
        print('\nCoupled Molecule state pairs:')
        print('')
        print('--------------------------------')
        for idx, (a, n) in enumerate(self._state_pairs):
            state = f'S{a},'
            print(f'   {idx:<3d}: ( {state:3s} n={n:<2d} ) ' + '*'*(idx in indices))
        print(' * Used for dynamics \n')
  
        return tuple(sorted(indices))
    

    def copy(self):
        new_copy = deepcopy(self)
        return new_copy
    
    def compute_all(self, all_energies, grads, nacs, dipole_matrix, dipole_matrix_grads, ref_eig_vecs=None):
        self.mol_energies = np.copy(all_energies)
        self.mol_gradients = np.copy(grads)
        self.mol_NACs = np.copy(nacs)
        self.mol_dipole_matrix_gradient = np.copy(dipole_matrix_grads)
        self.mol_dipole_matrix = np.copy(dipole_matrix)

        #   set hamiltonian, diagonalize, and compute needed gradients
        self.set_hamiltonian(all_energies, self.mol_dipole_matrix)
        self.set_hamiltonian_gradient(self.mol_gradients, self.mol_dipole_matrix, self.mol_dipole_matrix_gradient)
        self.diagonalize_H(ref_eig_vecs)
        self.calc_NA_coupling(self.mol_NACs)
        self.calc_eigen_value_gradient()
        # self.calc_eigen_vector_gradient()
        # self._calc_eigen_vector_gradient_reference()


    def calc_NA_coupling(self, mol_basis_NACs):
        basis_NACs = self.get_basis_NACs(mol_basis_NACs)
        return super().calc_NA_coupling(basis_NACs)

    def _calc_mu_dot_field(self, dipole_matrix):
        '''
            Parameters
            ----------
            dipole_matrix: np.ndarray (n_elec x n_elec x 3)
                Dipole matrix with diagonal elemnts being the dipoles of each
                state and the off-diagonal elements being the transition dipoles

            Returns
            -------
            mu_dot_field: np.ndarray (n_elec x n_elec)
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

        return mu_dot_field

    def _calc_dipole_self_energy(self, mu_dot_field):
        '''
            Parameters
            ----------
            mu_dot_field: np.ndarray (n_elec x n_elec)
                Dipole matrix dotted with the electric field direction
                or the norm of the dipole matrix if no field direction is provided.
                Computed in _calc_mu_dot_field() method

            Returns
            -------
            dipole_self: np.ndarray (n_elec x n_elec)
        '''
        n_elec = self._n_elec
        dipole_self = np.zeros((n_elec, n_elec))
        if self._use_DSE:
            for a in range(n_elec):
                for b in range(n_elec):
                    for gamma in range(n_elec):
                        dipole_self[a, b] += self.gc**2/self.omega_c * mu_dot_field[a, gamma]*mu_dot_field[gamma, b]
        return dipole_self

    def _calc_grad_mu_dot_field(self, dipole_matrix, dipole_grads):
        '''
            Parameters
            ----------
            dipole_grads: np.ndarray (n_elec x n_elec x 3*n_nuc, 3)
                Dipole matrix with diagonal elemnts being the dipoles of each
                state and the off-diagonal elements being the transition dipoles

            Returns
            -------
            grad_mu_dot_field: np.ndarray (n_elec x n_elec x 3*n_nuc)
        '''
        n_elec = self._n_elec
        
        grad_mu_dot_field = np.zeros((n_elec, n_elec, self.n_nuclei*3))
        if self._field_dir is not None:
            for a in range(n_elec):
                for b in range(n_elec):
                    grad_mu_dot_field[a, b] = np.einsum('ij,j->i', dipole_grads[a, b], self._field_dir)
                        
        #   Or, we assume that the field is always in the same direction as the dipole moments 
        else:
            for a in range(n_elec):
                for b in range(n_elec):
                    field = dipole_matrix[a, b]/np.linalg.norm(dipole_matrix[a, b])
                    grad_mu_dot_field[a, b] = np.einsum('ij,j->i', dipole_grads[a, b], field)

        return grad_mu_dot_field
    
    def _calc_grad_dipole_self_energy(self, mu_dot_field, grad_mu_dot_field):
        '''
            Parameters
            ----------
            mu_dot_field: np.ndarray (n_elec x n_elec)
                Dipole matrix dotted with the electric field direction.
                Computed in _calc_mu_dot_field() method.
            grad_mu_dot_field: np.ndarray (n_elec x n_elec x 3*n_nuc)
                Gradient of the dipole matrix dotted with the electric field direction.
                Computed in _calc_grad_mu_dot_field() method.

            Returns
            -------
            dipole_self_grad: np.ndarray (n_elec x n_elec, 3*n_nuc)
        '''
        n_elec = self._n_elec
        dipole_self_grad = np.zeros((n_elec, n_elec, self.n_nuclei*3))
        if self._use_DSE:
            pre_factor = self.gc**2/self.omega_c
            for a in range(n_elec):
                for b in range(n_elec):
                    for gamma in range(n_elec):
                        term1 = pre_factor * grad_mu_dot_field[a, gamma, :] * mu_dot_field[gamma, b]
                        term2 = pre_factor * mu_dot_field[a, gamma] * grad_mu_dot_field[gamma, b, :]
                        dipole_self_grad[a, b, :] += term1 + term2
        return dipole_self_grad

    def set_hamiltonian(self, energies, dipole_matrix):
        '''
            Evaluate the Hamiltonian elements. This must be done before
            computing any kind of gradients or eigenvalues.

            Parameters
            ----------
            energies: np.ndarray
                diagonal components of the hamiltonian
            dipole_matrix: np.ndarray (n_states x n_states)
                Dipole matrix with diagonal elemnts being the dipoles of each
                state and the off-diagonal elements being the transition dipoles
        '''
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

        mu_dot_field = self._calc_mu_dot_field(dipole_matrix)
        dipole_self = self._calc_dipole_self_energy(mu_dot_field)
        if not self._use_PDT:
            for a in range(n_elec):
                mu_dot_field[a, a] = 0.0

        print('Creating PF-Hamiltonian matrix')
        print(f'{self.gc=:.18f} {self.omega_c=:.18f}')
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

                print(f'  {i=:2d} {j=:2d} {H_en=:23.18f} {H_p=:23.18f} {H_en_p=:23.18f} {H_d=:23.18f} {mu_dot_field[a, b]=:23.18f}')

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

        mu_dot_field = self._calc_mu_dot_field(dipole_matrix)
        dipole_self = self._calc_dipole_self_energy(mu_dot_field)

        
        H_t = np.zeros((self._n_dim, self._n_dim))
        delta = np.eye(self._n_dim)
        for i, state_i in enumerate(self.state_pairs):
            for j, state_j in enumerate(self.state_pairs):
                a, m = state_i
                b, n = state_j

                H_en = energies[a]*delta[a, b]*delta[m, n]
                H_p = self.omega_c*(n + 0/2)*delta[a, b]*delta[m, n]
                H_d = dipole_self[a, b]*delta[m, n]

                if a < b:
                    H_en_p = self.gc * mu_dot_field[a, b] * sqrt(n+1)*delta[m, n+1]
                elif a > b:
                    H_en_p = self.gc * mu_dot_field[a, b] * sqrt(n)*delta[m, n-1]
                elif a == b and self._use_PDT:
                    H_en_p = self.gc * mu_dot_field[a, b] * (sqrt(n+1)*delta[m, n+1] + sqrt(n)*delta[m, n-1])
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

        mu_dot_field = self._calc_mu_dot_field(dipoles)
        grad_mu_dot_field = self._calc_grad_mu_dot_field(dipoles, dipole_grads)
        dipole_self_grad = self._calc_grad_dipole_self_energy(mu_dot_field, grad_mu_dot_field)

        if not self._use_PDT:
            for a in range(self._n_elec):
                mu_dot_field[a, a] = 0.0
                dipole_self_grad[a, a, :] = 0.0

        #   now form the gradient of the Hamiltonian matrix
        dH = np.zeros((self._n_dim, self._n_dim, self.n_nuclei*3))
        delta = np.eye(self._n_dim)
        for i, state_i in enumerate(self.state_pairs):
            for j, state_j in enumerate(self.state_pairs):
                a, m = state_i
                b, n = state_j

                dH_en = gradients[a, :]*delta[a, b]*delta[m, n]
                dH_d = dipole_self_grad[a, b, :]*delta[m, n]
                dH_en_p = self.gc * grad_mu_dot_field[a, b, :] * (sqrt(n)*delta[m, n-1] + sqrt(n+1)*delta[m, n+1])
                dH_p = 0.0
                dH[i, j] = dH_en + dH_p + dH_en_p + dH_d


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

        mu_dot_field = self._calc_mu_dot_field(dipoles)
        grad_mu_dot_field = self._calc_grad_mu_dot_field(dipoles, dipole_grads)
        dipole_self_grad = self._calc_grad_dipole_self_energy(mu_dot_field, grad_mu_dot_field)

        #   now form the gradient of the Hamiltonian matrix
        dH = np.zeros((self._n_dim, self._n_dim, self.n_nuclei*3))
        delta = np.eye(self._n_dim)
        for i, state_i in enumerate(self.state_pairs):
            for j, state_j in enumerate(self.state_pairs):
                a, m = state_i
                b, n = state_j

                dH_en = gradients[a, :]*delta[a, b]*delta[m, n]
                dH_p = 0.0
                dH_d = dipole_self_grad[a, b, :]*delta[m, n]

                if a < b:
                    dH_en_p = self.gc * grad_mu_dot_field[a, b, :] * sqrt(n+1)*delta[m, n+1]
                elif a > b:
                    dH_en_p = self.gc * grad_mu_dot_field[a, b, :] * sqrt(n)*delta[m, n-1]
                elif a == b and self._use_PDT:
                    dH_en_p = self.gc * grad_mu_dot_field[a, b, :] * (sqrt(n+1)*delta[m, n+1] + sqrt(n)*delta[m, n-1])
                else:
                    dH_en_p = 0.0

                dH_en_p = self.gc * grad_mu_dot_field[a, b, :] * (sqrt(n)*delta[m, n-1] + sqrt(n+1)*delta[m, n+1])

                dH[i, j, :] = dH_en + dH_p + dH_en_p + dH_d


        self.mol_gradients = gradients
        self.mol_dipole_matrix_gradient = dipole_grads
        self.dH = dH
        return dH

    def get_subset_properties(self):
        out_eigen_vals = self.eigen_vals[self._subset_state_indices]
        out_eigen_val_gradients = self.eigen_val_gradients[self._subset_state_indices][:, self._subset_state_indices]
        out_NACs = self.NACs[self._subset_state_indices][:, self._subset_state_indices, :]

        return (out_eigen_vals, out_eigen_val_gradients, out_NACs)
    
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
        scale_factor = coupling/(np.linalg.norm(trans_dipole)*sqrt(self.omega_c))
        g_c = scale_factor*sqrt(self.omega_c)
        self._gc = g_c

    def compute_adiabatic_states(self, energies, dipoles):

        # gc = _get_gc(coupling, omega_c, trans_dipole)
        self.hamiltonian = self._get_Nstate_hamiltonian(energies, dipoles)
        self.e_vals, self.e_vecs = self.diagonalize_H(self.hamiltonian)

