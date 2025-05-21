import unittest
import pandas
import numpy as np
import os
import json
import inspect
import pickle
import json

# sys.path.insert(1, os.path.join(os.path.dirname(pysces.__file__), '../../tests'))
# from tools import parse_xyz_data, assert_dictionary, cleanup, reset_directory
from pysces.qcRunners.TeraChem import TCJobBatch, TCJob
from pysces.h5file import H5File

from PySCESPol.Polariton import CoupledMolecule, format_combo_job_results

EV_2_AU = 1/27.2114079527

class pauil_fierz(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def setUp(self) -> None:
        pass

    def test_full_matrix(self):
        # reset_directory()
        this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        os.chdir(this_dir)

        with open(os.path.join(this_dir, 'job_results_list.pkl'), 'rb') as file:
            job_results = pickle.load(file)

        with open(os.path.join(this_dir, 'settings.json')) as file:
            settings = json.load(file)

        omega_c = settings['omega_c']
        grads = settings['grads']
        n_nuc = settings['n_nuc']
        field = settings['field']
        coupling = settings['coupling']
        tr_dipole_mag = settings['tr_dipole_mag']
        mol = CoupledMolecule(omega_c*EV_2_AU, grads, n_nuc, field)
        mol.set_gc_from_coupling(coupling*EV_2_AU, tr_dipole_mag)

        all_states = np.arange(0, max(mol.mol_grad_indices) + 1)
        all_energies, elecE, grads, nacs, dipole_matrix, dipole_matrix_grads = format_combo_job_results(job_results, all_states)
        mol.compute_all(all_energies, grads, nacs, dipole_matrix, dipole_matrix_grads)

        with open('ref_data_full_matrix.pkl', 'rb') as file:
            ref_data = pickle.load(file)

        np.testing.assert_allclose(ref_data['hamiltonian'], mol.hamiltonian, atol=1e-5, verbose=True, err_msg='key: hamiltonian')
        np.testing.assert_allclose(ref_data['hamiltonian_grads'], mol.dH, atol=1e-6, verbose=True, err_msg='key: hamiltonian_grads')
        np.testing.assert_allclose(ref_data['eigenvalues'], mol.eigen_vals, atol=1e-6, verbose=True, err_msg='key: eigenvalues')
        np.testing.assert_allclose(ref_data['eigenvectors'], mol.eigen_vecs, atol=1e-5, verbose=True, err_msg='key: eigenvectors')
        np.testing.assert_allclose(ref_data['eigenvalue_grads'], mol.eigen_val_gradients, atol=1e-6, verbose=True, err_msg='key: eigenvalue_grads')
        np.testing.assert_allclose(ref_data['NACs'], mol.NACs, atol=1e-3, verbose=True, err_msg='key: NACs')
        np.testing.assert_allclose(ref_data['dipole_matrix'], mol.mol_dipole_matrix, atol=1e-3, verbose=True, err_msg='key: dipole_matrix')
        np.testing.assert_allclose(ref_data['dipole_matrix_grads'], mol.mol_dipole_matrix_gradient, atol=1e-6, verbose=True, err_msg='key: dipole_matrix_grads')


                
if __name__ == '__main__':
    test = pauil_fierz()
    test.test_full_matrix()