import numpy as np
import matplotlib.pyplot as pl
from test_channels.channel_characteristics import rate_constant, compute_tau, boltzmann_fun


if __name__ == '__main__':

    v_range = np.arange(-100, 50, 0.1)

    # steady-state: Dickson
    vh_act_exp = -67.4
    k_act_exp = 12.66
    vh_inact_exp = -57.92
    k_inact_exp = 9.26
    act_slow_exp = boltzmann_fun(v_range, vh_act_exp, k_act_exp)
    act_fast_exp = boltzmann_fun(v_range, vh_inact_exp, k_inact_exp)

    # time constant Dickson
    a = -2.89 * 1e-3
    b = -0.445
    k = 24.02
    alpha_exp = rate_constant(v_range, a, b, k)
    a = 2.71 * 1e-2
    b = -1.024
    k = -17.4
    beta_exp = rate_constant(v_range, a, b, k)
    time_constant_fast_exp = compute_tau(alpha_exp, beta_exp)
    a = -3.18 * 1e-3
    b = -0.695
    k = 26.72
    alpha_exp = rate_constant(v_range, a, b, k)
    a = 2.16 * 1e-2
    b = -1.065
    k = -14.25
    beta_exp = rate_constant(v_range, a, b, k)
    time_constant_slow_exp = compute_tau(alpha_exp, beta_exp)

    # Fits
    g_m = 6.91043643e+01
    g_h = 1.57954292e+01
    a = 1.00000000e-04
    b = -5.62014719e+01
    k = -1.79960755e-01
    alpha_fit = rate_constant(v_range, a, b, k)
    a = -2.29611478e-01
    b = -7.12670950e+01
    k = 2.97234260e+01
    beta_fit = rate_constant(v_range, a, b, k)
    act_fit = alpha_fit / (alpha_fit+beta_fit)
    time_constant_fast_fit = compute_tau(alpha_fit, beta_fit)

    a = -1.00000000e-03
    b = -1.00000000e+02
    k = 2.38531987e+01
    alpha_fit = rate_constant(v_range, a, b, k)
    a = 1.00000000e-03
    b = 3.84100486e+01
    k = -2.08062432e+01
    beta_fit = rate_constant(v_range, a, b, k)

    inact_fit = alpha_fit / (alpha_fit+beta_fit)
    time_constant_slow_fit = compute_tau(alpha_fit, beta_fit)

    pl.figure()
    pl.plot(v_range, act_fast_exp, 'darkred', label='Fast activation (exp)', linewidth=1.5)
    pl.plot(v_range, act_fit, 'red', label='Fast activation (fit)', linewidth=1.5)
    pl.plot(v_range, act_slow_exp, 'darkblue', label='Slow activation (exp)', linewidth=1.5)
    pl.plot(v_range, inact_fit, '#a6bddb', label='Slow activation (fit)', linewidth=1.5)
    #pl.plot(v_range, boltzmann_fun(v_range, -64, 22), 'g', label='test', linewidth=1.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('Steady-state curve', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()

    pl.figure()
    pl.plot(v_range, time_constant_fast_exp, color='darkred', label='Fast activation (exp)', linewidth=1.5)
    pl.plot(v_range, time_constant_slow_exp, color='darkblue', label='Slow activation (exp)', linewidth=1.5)
    pl.plot(v_range, time_constant_fast_fit, color='red', label='Fast activation (fit)', linewidth=1.5)
    pl.plot(v_range, time_constant_slow_fit, color='#a6bddb', label='Slow activation (fit)', linewidth=1.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('Tau (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()