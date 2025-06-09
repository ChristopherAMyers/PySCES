from PySCESPol.Polariton import TCPolaritonRunner, CoupledMolecule, TCRunnerOptions
from qcelemental.models import Molecule

integrator = 'RK4'

#   number of atoms in the molecule
natom = 4
nel = 9 # number of electronic states
temp = 500 # simulation temperature in Kelvin

# Maximum propagation time (a.u.), one Runge-Kutta step (a.u.) (Only relevant for RK4)
tmax_rk4, Hrk4 = 10, 1.0

# Index of initially occupied electronic state
init_state = 2

# Restart request: 0 = no restart, 1 = restart
restart = 0
restart_file_in = 'restart.json'

#   TeraChem runner options
# tcr_host = ['10.1.1.166']*2
tcr_host = ['10.1.1.157', '10.1.1.166'][0]
tcr_port = [12341, 12341][0]
tcr_server_root = ['/home/cmyers7/scratch/single_servers/tmp1', '/home/cmyers7/scratch/single_servers/tmp2'][0]
# tcr_server_root = ['servers']*2
tcr_job_options = {
        'method': 'b3lyp',
        'basis': '3-21G',
        # 'basis': 'sto-2g',
        'charge': 0,
        'spinmult': 1,
        'closed_shell': True,
        'restricted': True,
        'precision': 'mixed',
        'convthre': 1E-6,
        'sphericalbasis': 'yes',

        #   TD-DFT
        'cis': 'yes',
        'cisnumstates': 4,
        'cisrestart': 'cis_restart',
        'cisconvtol': 1e-5,
        'cismaxiter': 500,

        # 'cischarges': 'yes',
        # 'resp': 'yes',
}
tcr_state_options = {
    'max_state': 4
}
tcr_spec_job_opts = {
    'gradient_1': {
        'cisdipolederiv': 'yes',
        'dipolederivative': 'yes',
        'cpcisiter': 1000,
    },
    'gradient_2': {
        'cistransdipolederiv': 'yes',
        'cpcisiter': 1000,
    }
}


# Terachem files
fname_tc_xyz      = "freq/mol.xyz"
fname_tc_geo_freq = "freq/Geometry.frequencies.dat"
fname_tc_redmas   = "freq/Reduced.mass.dat"
fname_tc_freq     = "freq/Frequencies.dat"

input_seed = 123456

hdf5_logging = True
logging_mode = 'w'
state_labels = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
init_state = 4

EV_2_AU = 1/27.2114079527
coupled_mol = CoupledMolecule(9.56740788*EV_2_AU, [1,2,3,4], 4, [0.5889,    -0.3378,    -0.0286])
coupled_mol.set_gc_from_coupling(0.3*EV_2_AU, 0.6795)
mol = Molecule.from_file('freq/mol.xyz')
tc_runner_opts = TCRunnerOptions()
tc_runner_opts.host = tcr_host
tc_runner_opts.port = tcr_port
tc_runner_opts.job_options = tcr_job_options
tc_runner_opts.state_options = {'grads': [1, 2, 3, 4]}
tc_runner_opts.server_root = tcr_server_root
tc_runner_opts.spec_job_opts = tcr_spec_job_opts
QC_RUNNER = TCPolaritonRunner(coupled_mol, mol.symbols, tc_runner_opts)
# extra_loggers = [QC_RUNNER.polariton_logger, QC_RUNNER.tc_logger]
mol_input_format = 'terachem'
QC_RUNNER.set_print_level(0)
