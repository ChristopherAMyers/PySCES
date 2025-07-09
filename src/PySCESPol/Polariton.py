import numpy as np
from numpy.linalg import norm
from numpy import sqrt, abs
from dataclasses import dataclass
from pysces.qcRunners import TeraChem
from pysces.qcRunners.TeraChem import TCRunner, TCJob, TCJobBatch, TCRunnerOptions, format_combo_job_results
from pysces.fileIO import LoggerData, H5File
from pysces.subroutines import SignFlipper
from qcelemental.models import Molecule
from qcelemental.periodic_table import periodictable as pt

from pysces.qcRunners.TeraChem import TCJobsLogger
from pysces.h5file import h5py

import pickle
import os
from copy import deepcopy
import warnings
from typing import Literal
from time import time
import json
from collections import deque
from collections.abc import Iterable, Sequence

from . import NumDeriv as numD
from .Interpolation import DipoleMatrixTracker
from .CoupledMolecule import CoupledMolecule

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

class PolaritonLogger():
    name = 'polariton'
    def __init__(self) -> None:
        self._h5_file: H5File = None
        self._h5_group: h5py.Group = None
        self._initialized = False
        self._next_dataset: dict[str, np.ndarray] = {}
        self._labels: dict[str,list[str]] = {}

    def setup(self, h5_file: H5File):
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

    def write(self, time: float):
        if not self._initialized:
            self._initialize()
        H5File.append_dataset(self._h5_group['time'], time)
        for key, data in self._next_dataset.items():
            H5File.append_dataset(self._h5_group[key], data)


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
    
    def save_restart(self):
        from .Serialization import TCPolaritonRunnerSerialize
        restart_data = TCPolaritonRunnerSerialize(self)
        return restart_data

    def load_restart(self, restart_data):
        from .Serialization import TCPolaritonRunnerDeserialize
        TCPolaritonRunnerDeserialize(restart_data, self)

    def set_logger_file(self, h5_file: H5File):
        super().set_logger_file(h5_file)
        self.polariton_logger.setup(h5_file)


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
        dipoles, tr_dipoles = self._check_new_dipole_grads_to_run()
        job_batch = self.create_jobs(geom, False, self._grads, self._NACs, dipoles, tr_dipoles)
        job_batch = self._send_jobs_to_clients(job_batch)

        #   step 2
        if self._interpolate_grads or self._interpolate_NACs:
            new_grads, new_nacs = self._check_new_grads_nacs_to_run(job_batch)
            new_dipole_grads, new_tr_dipole_grads = self._check_new_dipole_grads_to_run()
            job_batch_2 = self.create_jobs(geom, False, new_nacs, new_grads, new_dipole_grads, new_tr_dipole_grads)
            job_batch_2 = self._send_jobs_to_clients(job_batch_2)

            #   step 3: combine both batches
            job_batch.jobs += job_batch_2.jobs

        self._log_jobs(job_batch, self._frame_counter)
        self.compute_coupled_mol_properties(job_batch.results_list)
        self.log_timestep()
        self.print_results()
        self._finalize_frame(job_batch)

        return self.get_pysces_outputs()
        
    def _check_new_dipole_grads_to_run(self):
        if self.coupled_mol._use_RWA and not self.coupled_mol._use_DSE:
            dipoles = ()
        else:
            dipoles = np.arange(min(self._grads), max(self._grads) + 1, dtype=int).tolist()

        tr_dipoles = [(0, x) for x in range(1, max(self._grads) + 1)] 

        if not self._run_dipole_derivative_interpolation:
            return dipoles, tr_dipoles
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

    def get_pysces_outputs(self):
        # #   TODO: Compute transition dipoles!!!
        mol = self.coupled_mol

        # out_eigen_vals, out_eigen_val_grads, out_NACs = self.coupled_mol.get_subset_properties()
        
        if (0, 0) in mol.state_pairs:
            out_all_energies = mol.eigen_vals.copy()
        else:
            out_all_energies = np.append(np.min(mol.mol_energies), mol.eigen_vals)
        out_eigen_vals = mol.eigen_vals
        out_eigen_val_grads = mol.eigen_val_gradients
        out_NACs = mol.NACs

        if self._rk4_inteprolation:
            self._previous_pysces_outputs = None
            raise NotImplementedError('RK4 interpolation not yet implemented')
        else:
            return (out_all_energies, out_eigen_vals, out_eigen_val_grads, out_NACs, None, None)

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
        self.polariton_logger.write(self._frame_counter)

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

