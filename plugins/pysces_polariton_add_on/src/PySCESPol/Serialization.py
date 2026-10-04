from .Polariton import SignFlipper, TCPolaritonRunner
from .Interpolation import DipoleMatrixTracker
from .AdiabaticStates import AdiabaticStates
from .CoupledMolecule import CoupledMolecule
import numpy as np
from pysces.serialization import TCRunner_Deserialize, TCRunner_Serialize
from pysces.qcRunners.TeraChem import TCJob
from collections import deque

# Global registry to keep track of class proxies
proxy_registry = {}

def register_proxy(class_name, proxy_class):
    """Register a proxy for a given class name."""
    proxy_registry[class_name] = proxy_class

def get_proxy(class_name):
    """Retrieve the proxy class for a given class name."""
    return proxy_registry.get(class_name)

# Base class for proxies
class Proxy:
    def serialize(self, obj):
        raise NotImplementedError

    def deserialize(self, data):
        raise NotImplementedError

class _Serializer:
    @staticmethod
    def serialize(obj):
        serialized_data = {"class": obj.__class__.__name__}

        for attr_name, attr_value in obj.__dict__.items():
            class_name = attr_value.__class__.__name__
            proxy = get_proxy(class_name)
            if proxy:
                serialized_data[attr_name] = proxy.serialize(attr_value)
            else:
                serialized_data[attr_name] = attr_value  # Primitive data types
        return serialized_data

    @staticmethod
    def deserialize(data, obj=None):
        # Get the class name from the data
        class_name = data["class"]

        # Retrieve the proxy for the class
        proxy = get_proxy(class_name)
        if proxy:
            # Use the provided object, or create a new one if obj is None
            obj = proxy.deserialize(data, obj)
        else:
            raise ValueError(f"No proxy registered for class: {class_name}")

        # Deserialize each member variable using its proxy
        for attr_name, attr_value in data.items():
            if attr_name == "class":
                continue  # Skip the class name

            if isinstance(attr_value, dict) and "class" in attr_value:
                # If it's a nested class, deserialize it
                current_value = getattr(obj, attr_name, None)
                setattr(obj, attr_name, _Serializer.deserialize(attr_value, current_value))
            else:
                setattr(obj, attr_name, attr_value)

        return obj


class SignFlipperProxy(Proxy):
    def __init__(self, obj: SignFlipper) -> None:
        self.obj = obj

    def serialize(self) -> dict:
        return {
            'class': 'SignFlipper',
            'n': self.obj.n,
            'n_signs': self.obj.n_signs,
            'n_dof': self.obj.n_dof,
            'name': self.obj.name
        }
    
    def deserialize(self, data: dict, obj: SignFlipper) -> None:
        obj.n = data['n']
        obj.n_signs = data['n_signs']
        obj.n_dof = data['n_dof']
        obj.name = data['name']

class CoupledMoleculeProxy(Proxy):
    pass

class DipoleMatrixTrackerProxy(Proxy):
    def __init__(self, obj: DipoleMatrixTracker) -> None:
        self.obj = obj

    def serialize(self) -> dict:
        return {
            'class': 'DipoleMatrixTracker',
            'order': self.obj.order,
            'dipole_matrix': self.obj.dipole_matrix,
            'dipole_matrix_gradient': self.obj.dipole_matrix_gradient,
            'dipole_matrix_hessian': self.obj.dipole_matrix_hessian
        }
    
    def deserialize(self, data: dict, obj: DipoleMatrixTracker) -> None:
        obj.dipole_matrix = data['dipole_matrix']
        obj.dipole_matrix_gradient = data['dipole_matrix_gradient']
        obj.dipole_matrix_hessian = data['dipole_matrix_hessian']

#   Register all proxies
register_proxy('SignFlipper', SignFlipperProxy)

def AdibaticStatesSerialize(adibatic_states: AdiabaticStates) -> dict:
    """Serialize an AdiabaticStates object."""
    serialized_data = {
        'class': adibatic_states.__class__.__name__,
        'n_states': adibatic_states.n_states,
        'n_nuclei': adibatic_states.n_nuclei,
        'hamiltonian': adibatic_states.hamiltonian.tolist(),
        'dH': adibatic_states.dH.tolist(),
        'eigen_vals': adibatic_states.eigen_vals.tolist(),
        'eigen_vecs': adibatic_states.eigen_vecs.tolist(),
        'NACs': adibatic_states.NACs.tolist(),
        'eigen_val_gradients': adibatic_states.eigen_val_gradients.tolist(),
        'eigen_vec_gradients': adibatic_states.eigen_vec_gradients.tolist(),
        'diagonalized': adibatic_states._diagonalized
    }
    return serialized_data

def AdiabaticStatesDeserialize(data: dict, adiabatic_states: AdiabaticStates) -> AdiabaticStates:
    """Deserialize an AdiabaticStates object."""

    adiabatic_states._n_states = data['n_states']
    adiabatic_states._n_nuclei = data['n_nuclei']
    adiabatic_states._hamiltonian = np.array(data['hamiltonian'])
    adiabatic_states._dH = np.array(data['dH'])
    adiabatic_states._diagonalized = data['diagonalized']

    adiabatic_states.eigen_vals = np.array(data['eigen_vals'])
    adiabatic_states.eigen_vecs = np.array(data['eigen_vecs'])
    adiabatic_states.NACs = np.array(data['NACs'])
    adiabatic_states.eigen_val_gradients = np.array(data['eigen_val_gradients'])
    adiabatic_states.eigen_vec_gradients = np.array(data['eigen_vec_gradients'])
    
    return adiabatic_states

def CoupledMoleculeSerialize(coupled_mol: CoupledMolecule) -> dict:
    """Serialize a CoupledMolecule object."""
    serialized_data = AdibaticStatesSerialize(coupled_mol)  # Serialize inherited attributes
    serialized_data.update({
        'class': coupled_mol.__class__.__name__,
        '_omega_c': coupled_mol._omega_c,
        '_gc': coupled_mol._gc,
        '_field_dir': coupled_mol._field_dir.tolist() if coupled_mol._field_dir is not None else None,
        '_use_DSE': coupled_mol._use_DSE,
        '_use_RWA': coupled_mol._use_RWA,
        '_use_PDT': coupled_mol._use_PDT,
        '_state_pairs': coupled_mol._state_pairs,
        '_subset_state_indices': coupled_mol._subset_state_indices,
        'mol_energies': coupled_mol.mol_energies.tolist(),
        'mol_gradients': coupled_mol.mol_gradients.tolist(),
        'mol_NACs': coupled_mol.mol_NACs.tolist(),
        'mol_dipole_matrix': coupled_mol.mol_dipole_matrix.tolist(),
        'mol_dipole_matrix_gradient': coupled_mol.mol_dipole_matrix_gradient.tolist(),
    })
    return serialized_data

def CoupledMoleculeDeserialize(data: dict, coupled_mol: CoupledMolecule) -> CoupledMolecule:
    """Deserialize a CoupledMolecule object."""
    coupled_mol = AdiabaticStatesDeserialize(data, coupled_mol)  # Deserialize inherited attributes
    coupled_mol._omega_c = data['_omega_c']
    coupled_mol._gc = data['_gc']
    coupled_mol._field_dir = np.array(data['_field_dir']) if data['_field_dir'] is not None else None
    coupled_mol._use_DSE = data['_use_DSE']
    coupled_mol._use_RWA = data['_use_RWA']
    coupled_mol._use_PDT = data['_use_PDT']
    coupled_mol._state_pairs = tuple(tuple(pair) for pair in data['_state_pairs'])
    coupled_mol._subset_state_indices = tuple(data['_subset_state_indices'])
    coupled_mol.mol_energies = np.array(data['mol_energies'])
    coupled_mol.mol_gradients = np.array(data['mol_gradients'])
    coupled_mol.mol_NACs = np.array(data['mol_NACs'])
    coupled_mol.mol_dipole_matrix = np.array(data['mol_dipole_matrix'])
    coupled_mol.mol_dipole_matrix_gradient = np.array(data['mol_dipole_matrix_gradient'])
    return coupled_mol

def TCPolaritonRunnerSerialize(runner: TCPolaritonRunner) -> dict:
    """Serialize a TCPolaritonRunner object."""
    serialized_data = TCRunner_Serialize(runner)
    serialized_data.update({
        'class': runner.__class__.__name__,
        'coupled_molecule': CoupledMoleculeSerialize(runner.coupled_mol),
        'tc_opts': runner._base_options,
        'max_wait': runner._max_wait,
        'prev_ref_job': runner._prev_ref_job.results if runner._prev_ref_job else None,
        'frame_counter': runner._frame_counter,
        'run_dipole_derivative_interpolation': runner._run_dipole_derivative_interpolation,
        'print_level': runner._print_level,

        'momentum_history_time': [x[0] for x in runner._momentum_history],
        'momentum_history_vals': [np.array(x[1]).tolist() for x in runner._momentum_history],

        'position_history_time': [x[0] for x in runner._position_history],
        'position_history_vals': [np.array(x[1]).tolist() for x in runner._position_history],

        'dipole_matrix_history_time': [x[0] for x in runner._dipole_matrix_history],
        'dipole_matrix_history_vals': [np.array(x[1]).tolist() for x in runner._dipole_matrix_history],
    })
    return serialized_data

def TCPolaritonRunnerDeserialize(data: dict, runner: TCPolaritonRunner):
    """Deserialize a TCPolaritonRunner object."""
    TCRunner_Deserialize(data, runner)
    runner.coupled_mol = CoupledMoleculeDeserialize(data['coupled_molecule'], runner.coupled_mol)
    runner._base_options = data['tc_opts']
    runner._max_wait = data['max_wait']
    runner._prev_ref_job = TCJob(results=data['prev_ref_job']) if data['prev_ref_job'] else None
    runner._frame_counter = data['frame_counter']
    runner._run_dipole_derivative_interpolation = data['run_dipole_derivative_interpolation']
    runner._print_level = data['print_level']

    for i in range(len(data['momentum_history_time'])):
        time = data['momentum_history_time'][i]
        vals = data['momentum_history_vals'][i]
        runner._momentum_history.append((time, np.array(vals)))

    for i in range(len(data['position_history_time'])):
        time = data['position_history_time'][i]
        vals = data['position_history_vals'][i]
        runner._position_history.append((time, np.array(vals)))

    for i in range(len(data['dipole_matrix_history_time'])):
        time = data['dipole_matrix_history_time'][i]
        vals = data['dipole_matrix_history_vals'][i]
        runner._dipole_matrix_history.append((time, np.array(vals)))
    
