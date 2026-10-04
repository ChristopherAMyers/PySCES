import numpy as np
from collections import deque


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
