import numpy as np

__author__ = 'caro'


class IonChannel:
    def __init__(self, g_max, ep, n_gates, power_gates, inf_gates, tau_gates):
        self.g_max = g_max  # (S/cm2)
        self.ep = ep  # (mV) equilibrium potential
        self.n_gates = n_gates
        self.power_gates = np.array(power_gates)
        self.inf_gates = inf_gates  # function of V
        self.tau_gates = tau_gates  # function of V

    def compute_current(self, vs, p_gates):
        if self.n_gates == 0:
            return self.g_max * (vs - self.ep)
        else:
            return self.g_max * np.prod(p_gates**self.power_gates) * (vs - self.ep)  # (mA/cm2)

    def init_gates(self, v0, p_gates0=None):
        if p_gates0 is not None:
            return p_gates0
        else:
            return self.inf_gates(v0)

    def derivative_gates(self, vs, p_gate, n):
        return (self.inf_gates(vs)[n] - p_gate) / self.tau_gates(vs)[n]



# TODO: how to store ep global, for all ion channels accessible