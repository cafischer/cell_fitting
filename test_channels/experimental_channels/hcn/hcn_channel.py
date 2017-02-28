from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from test_channels.channel_characteristics import boltzmann_fun, rate_constant, tau


if __name__ == '__main__':

    # Dickson:

    # plot activation curve
    v_range = np.arange(-120, 0, 0.1)

    # fast
    vh_fast = -67.4
    k_fast = 12.66

    # slow
    vh_slow = -57.92
    k_slow = 9.26

    curve_act_fast = boltzmann_fun(v_range, vh_fast, k_fast)
    curve_act_slow = boltzmann_fun(v_range, vh_slow, k_slow)

    pl.figure()
    pl.plot(v_range, curve_act_fast, color='lightblue', label='fast')
    pl.plot(v_range, curve_act_slow, color='darkblue', label='slow')
    pl.xlabel('V (mV)')
    pl.ylabel('G (normalized)')
    pl.legend()
    #pl.show()

    # plot time constants
    a = -2.89 * 1e-3
    b = -0.445
    k = 24.02
    alpha = rate_constant(v_range, a, b, k)
    a = 2.71 * 1e-2
    b = -1.024
    k = -17.4
    beta = rate_constant(v_range, a, b, k)
    time_constant_fast = tau(alpha, beta)

    a = -3.18 * 1e-3
    b = -0.695
    k = 26.72
    alpha = rate_constant(v_range, a, b, k)
    a = 2.16 * 1e-2
    b = -1.065
    k = -14.25
    beta = rate_constant(v_range, a, b, k)
    time_constant_slow = tau(alpha, beta)

    pl.figure()
    pl.plot(v_range, time_constant_fast, color='lightblue', label='fast')
    pl.plot(v_range, time_constant_slow, color='darkblue', label='slow')
    pl.xlabel('V (mV)')
    pl.ylabel('Tau (ms)')
    pl.legend()
    pl.show()