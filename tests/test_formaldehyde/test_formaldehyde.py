import unittest
import pandas
import numpy as np
import os
import json
import inspect
import pysces
import sys

sys.path.insert(1, os.path.join(os.path.dirname(pysces.__file__), '../../tests'))
from tools import parse_xyz_data, assert_dictionary, cleanup, reset_directory
from pysces.qcRunners.TeraChem import TCJobBatch, TCJob
import pysces.qcRunners.TeraChem as TC
from pysces.h5file import H5File


class TEST_Formaldehyde(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def setUp(self) -> None:
        #   reset class variables
        TCJobBatch._TCJobBatch__batch_counter = 0
        TCJob._TCJob__job_counter = 0

        #   Load the reference nacs and use their values as sign references
        #   This will be later fixed with overlaps
        ref_nacs = np.loadtxt('ref_logs/nac.txt', skiprows=3, max_rows=18)
        pysces.SignFlipper._debug = True
        pysces.SignFlipper._ref_nacs = ref_nacs

    def test_jobs(self):
        # reset_directory()
        this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        os.chdir(this_dir)

        TC._DEBUG_LOAD_TRAJ = 'tc_traj_data.pkl'
        pysces.reset_settings()
        pysces.options.input_local_settings()
        pysces.options.make_logging_dir()
        pysces.run_simulation()

        data_tst = H5File('logs.h5')
        with open('ref_logs/polariton.json') as file:
            data_ref = json.load(file)

        #   compare electornic data
        for k in data_tst['electronic']:
            for frame in range(len(data_tst['electronic'][k])):
                np.testing.assert_allclose(data_tst['electronic'][k][frame], data_ref['electronic'][k][frame],
                                   atol=1e-3, verbose=True,
                                   err_msg=f'key: electronic/{k} frame: {frame}')

        #   compare polariton data
        for k in data_tst['polariton']:
            np.testing.assert_allclose(data_tst['polariton'][k], data_ref['polariton'][k],
                                       atol=1e-3, verbose=True,
                                       err_msg=f'key: polariton/{k}')


        #   check simple panda readable data
        for file in ['corr.txt', 'electric_pq.txt', 'energy.txt', 'grad.txt', 'nac.txt']:
            data_ref = pandas.read_csv(f'ref_logs/{file}', sep='\s+', comment='#')
            data_tst = pandas.read_csv(f'logs/{file}', sep='\s+', comment='#')
            for key in data_ref:
                np.testing.assert_allclose(data_tst[key], data_ref[key], 
                                           atol=1e-3, verbose=True,
                                           err_msg=f'file: {file} key: {key}')
        
        #   check data in xyz formats
        for file in ['nuc_geo.xyz', 'nuclear_P.txt']:
            data_ref = parse_xyz_data(f'ref_logs/{file}')
            data_tst = parse_xyz_data(f'logs/{file}')
            for frame, (frame_tst, frame_ref) in enumerate(zip(data_tst, data_ref)):
                np.testing.assert_equal(frame_tst['atoms'], frame_ref['atoms'])
                np.testing.assert_allclose(frame_tst['positions'], frame_ref['positions'],
                                           atol=1e-4, verbose=True,
                                           err_msg=f'file: {file}; frame {frame}')
        
        #   check checkpoint files
        with open('restart_end.json') as file:
            restart_ref = json.load(file)
        with open('restart.json') as file:
            restart_tst = json.load(file)
        for k in ['TCJobBatch__batch_counter', 'TCJob__job_counter']:
            restart_ref.pop(k, None)
            restart_tst.pop(k, None)
        assert_dictionary(self, restart_ref, restart_tst, atol=1e-3)

        cleanup()
        os.remove('logs.h5')

                
if __name__ == '__main__':
    test = TEST_Formaldehyde()
    test.test_jobs()