from PySCESPol.Polariton import SignFlipper, DipoleMatrixTracker, CoupledMolecule, TCPolaritonRunner

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

class Serializer:
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
                setattr(obj, attr_name, Serializer.deserialize(attr_value, current_value))
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