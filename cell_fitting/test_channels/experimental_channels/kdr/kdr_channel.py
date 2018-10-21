import numpy as np
import matplotlib.pyplot as pl
from cell_fitting.test_channels.channel_characteristics import plot_activation_curves, boltzmann_fun


if __name__ == '__main__':

    v_range = np.arange(-150, 50, 0.1)

    # activation: Eder
    vh_act = -4.8
    k_act = -11
    vh_std_act = 0
    k_std_act = 0

    # inactivation: Eder
    vh_inact = -74.3
    k_inact = 15
    vh_std_inact = 0
    k_std_inact = 0

    curve_act = boltzmann_fun(v_range, vh_act, k_act)
    curve_inact = boltzmann_fun(v_range, vh_inact, k_inact)
    # inactivation is always above 0.5 -> cannot be modelled by Boltzmann between 0 and 1! -> so use smaller range
    curve_inact = curve_inact / 2 + 0.5

    pl.figure()
    pl.plot(v_range, curve_act, color='r', label='Activation')
    pl.plot(v_range, curve_inact, color='b', label='Inactivation')
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('G (normalized)', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()