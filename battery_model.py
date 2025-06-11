import numpy as np
import scipy.optimize as opt

class BatteryModel:
    def __init__(self, R=8.314, T=298, Ec=50, Cr=1.0, Ed=50, Q=100, Dr=1.0):
        self.R = R  # Universal gas constant (J/mol-K)
        self.T = T  # Operating temperature (Kelvin)
        self.Ec = Ec  # Energy processed during charging (kWh)
        self.Cr = Cr  # Charge rate (normalized)
        self.Ed = Ed  # Energy processed during discharging (kWh)
        self.Q = Q  # Battery capacity (Ah)
        self.Dr = Dr  # Discharge rate
        self.cycles = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000])  # Cycles
        self.capacity = np.array([100, 95, 90, 85, 80, 75, 70])  # Percentage of original capacity

    def _R1(self, a, soc, rr):
        return (a[0] + a[1]*rr + a[2]*(rr**2)) * np.exp(-a[3]*soc) + (a[4] + a[5]*rr + a[6]*(rr**2))

    def _R2(self, a, soc, rr):
        return (a[7] + a[8]*rr + a[9]*(rr**2)) * np.exp(-a[10]*soc) + (a[11] + a[12]*rr + a[13]*(rr**2))

    def _C1(self, a, soc, rr):
        term1 = -(a[14] + a[15]*rr + a[16]*(rr**2)) * np.exp(-a[17]*soc)
        term2 = (a[18] + a[19]*soc + a[20]*(rr**2))
        return max(term1 + term2, 1e-6)

    def _V0(self, a, soc, rr):
        return (a[21] + a[22]*rr + a[23]*(rr**2)) * np.exp(-a[24]*soc) + (a[25] + a[26]*soc + a[27]*(soc**2) + a[28]*(soc**3)) - a[29]*rr + a[30]*(rr**2)
    
    def terminal_voltage(self, I, t, C, params, soc, isCharging):
        rr = 1.0
        if isCharging:
            rr == self.Cr
        else:
            rr = self.Dr
        R1 = self._R1(params, soc, rr)
        R2 = max(self._R2(params, soc, rr), 1e-3)
        C1 = max(self._C1(params, soc, rr), 1e-3)
        V0 = self._V0(params, soc, rr)

        voltage = ((self.Q/C + I*R2) * np.exp(-(t / (R2 * C1)))) + V0 - (I*(R1 + R2))
        return voltage
    

    # Capacity degradation model function
    def degradation_model(self, params, x):
        A, B, C, D = params  # Adjustable coefficients
        return 100 - (A * np.exp((C * self.Ec * self.Cr) / (self.R * self.T)) + B * np.exp((D * self.Ed * self.Q * self.Dr) / (self.R * self.T))) * x

    # Error function for optimization
    def _error_function(self, params):
        predicted = self.degradation_model(params, self.cycles)
        return np.sum((predicted - self.capacity) ** 2)

    def get_optimal_ABCD_params(self):
        # Optimize A, B, C, D to best fit the battery data
        initial_guess = [0.1, 0.1, 1.2, 1.5]  # Starting estimates
        return opt.minimize(self._error_function, initial_guess, method="L-BFGS-B")
    
    def get_predicted_voltages(self, df, C, params, isCharging):
        return df.apply(
            lambda row: self.terminal_voltage(
                I=row["Battery Current(A)"],
                t=row["time_diff_sec"],
                C=C,
                params=params,
                soc=row["SoC"],
                isCharging=isCharging
            ),
            axis=1
        )
    
