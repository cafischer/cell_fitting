import numpy as np
import matplotlib.pyplot as pl
from test_channels.channel_characteristics import plot_activation_curves, rate_constant, tau


if __name__ == '__main__':

    v_range = np.arange(-100, -10, 0.1)

    # activation: Magistretti
    vh_act = -44.4
    k_act = -5.2
    vh_std_act = 0
    k_std_act = 0

    # inactivation: Magistretti
    vh_inact = -48.8
    k_inact = 10
    vh_std_inact = 0
    k_std_inact = 0

    plot_activation_curves(v_range, vh_act, k_act, vh_std_act, k_std_act, vh_inact, k_inact, vh_std_inact, k_std_inact)

    # plot time constants
    a = -2.88 * 1e-3
    b = -4.9 * 1e-2
    k = 4.63
    alpha = rate_constant(v_range, a, b, k)
    a = 6.94 * 1e-3
    b = 0.447
    k = -2.63
    beta = rate_constant(v_range, a, b, k)
    time_constant_inact = tau(alpha, beta)

    pl.figure()
    pl.plot(v_range, time_constant_inact, color='b', label='inactivation')
    pl.xlabel('V (mV)')
    pl.ylabel('Tau (s)')
    pl.legend()
    pl.show()