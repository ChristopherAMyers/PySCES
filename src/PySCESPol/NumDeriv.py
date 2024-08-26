import numpy as np

class NumericalDerivative:
    def __init__(self, n_points, step_size):
        """
        Initialize the NumericalDerivative class.

        Parameters:
        -----------
        n_points : int
            The number of points used for the numerical derivative computation.
        step_size : float
            The step size used for the derivative computation.
        """
        self.n_points = n_points
        self.step_size = step_size
        self.buffer = []

    def add_entry(self, entry):
        """
        Add a new entry to the buffer.

        Parameters:
        -----------
        entry : list or numpy array
            A new data entry to be added to the buffer.
        """
        self.buffer.append(entry)

    def compute_derivative(self):
        """
        Compute the numerical derivative for all indices in the flattened buffer.

        Returns:
        --------
        grad : numpy array
            A numpy array containing the computed numerical derivatives for each index.
        """

        num_entries = len(self.buffer)//(self.n_points - 1)
        self.buffer = np.array(self.buffer)
        grad = np.zeros((num_entries,) + self.buffer[0].shape)

        for i in range(num_entries):
            start_idx = i*(self.n_points-1)
            end_idx = (i+1)*(self.n_points-1)
            sub_buffer = self.buffer[start_idx:end_idx]

            if self.n_points == 3:
                grad[i] -= sub_buffer[0]
                grad[i] += sub_buffer[1]
                grad[i] /= 2.0 * self.step_size
            elif self.n_points == 5:
                grad[i] += sub_buffer[0]
                grad[i] -= 8.0 * sub_buffer[1]
                grad[i] += 8.0 * sub_buffer[2]
                grad[i] -= sub_buffer[3]
                grad[i] /= (12.0 * self.step_size)
            elif self.n_points == 7:
                grad[i] -= sub_buffer[0]
                grad[i] += 9.0 * sub_buffer[1]
                grad[i] -= 45.0 * sub_buffer[2]
                grad[i] += 45.0 * sub_buffer[3]
                grad[i] -= 9.0 * sub_buffer[4]
                grad[i] += sub_buffer[5]
                grad[i] /= 60.0 * self.step_size
            elif self.n_points == 9:
                grad[i] += 3.0 * sub_buffer[0]
                grad[i] -= 32.0 * sub_buffer[1]
                grad[i] += 168.0 * sub_buffer[2]
                grad[i] -= 672.0 * sub_buffer[3]
                grad[i] += 672.0 * sub_buffer[4]
                grad[i] -= 168.0 * sub_buffer[5]
                grad[i] += 32.0 * sub_buffer[6]
                grad[i] -= 3.0 * sub_buffer[7]
                grad[i] /= 840.0 * self.step_size
            else:
                raise ValueError("Unsupported NPoints value")
        
        return grad

def generate_coords(ref_coords, n_points, dx):
    shift_multiples = (np.arange(n_points) - n_points//2).tolist()
    shift_multiples.pop(n_points//2)
    if len(ref_coords.shape) < 2 or ref_coords.shape[-1] != 3:
        raise ValueError("`ref_coords` must be of dimension (Nx3)")
    
    indicies = []
    new_geoms = []
    for n in range(len(ref_coords)):
        for i in [0, 1, 2]:
            shift_multiples = (np.arange(n_points) - n_points//2).tolist()
            shift_multiples.pop(n_points//2)
            for j in shift_multiples:
                indicies.append((n, i, j))

        for n, i, j in indicies:
            new_g = np.copy(ref_coords)
            new_g[n, i] += j*dx
            new_geoms.append(new_g)
        
    return new_geoms

def compute(values, n_points, step_size, transpose_seq=None):
        """
        Compute the numerical derivative for all indices in the flattened buffer.

        Returns:
        --------
        grad : numpy array
            A numpy array containing the computed numerical derivatives for each index.
        """

        num_entries = len(values)//(n_points - 1)
        values = np.array(values)
        grad = np.zeros((num_entries,) + values[0].shape)

        for i in range(num_entries):
            start_idx = i*(n_points-1)
            end_idx = (i+1)*(n_points-1)
            sub_values = values[start_idx:end_idx]

            if n_points == 3:
                grad[i] -= sub_values[0]
                grad[i] += sub_values[1]
                grad[i] /= 2.0 * step_size
            elif n_points == 5:
                grad[i] += sub_values[0]
                grad[i] -= 8.0 * sub_values[1]
                grad[i] += 8.0 * sub_values[2]
                grad[i] -= sub_values[3]
                grad[i] /= (12.0 * step_size)
            elif n_points == 7:
                grad[i] -= sub_values[0]
                grad[i] += 9.0 * sub_values[1]
                grad[i] -= 45.0 * sub_values[2]
                grad[i] += 45.0 * sub_values[3]
                grad[i] -= 9.0 * sub_values[4]
                grad[i] += sub_values[5]
                grad[i] /= 60.0 * step_size
            elif n_points == 9:
                grad[i] += 3.0 * sub_values[0]
                grad[i] -= 32.0 * sub_values[1]
                grad[i] += 168.0 * sub_values[2]
                grad[i] -= 672.0 * sub_values[3]
                grad[i] += 672.0 * sub_values[4]
                grad[i] -= 168.0 * sub_values[5]
                grad[i] += 32.0 * sub_values[6]
                grad[i] -= 3.0 * sub_values[7]
                grad[i] /= 840.0 * step_size
            else:
                raise ValueError("Unsupported NPoints value")
        
        if transpose_seq is not None:
            grad = grad.transpose(transpose_seq)

        return grad