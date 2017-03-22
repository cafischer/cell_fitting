import numpy as np
import matplotlib.pyplot as pl
from test_channels.channel_characteristics import rate_constant, compute_tau, boltzmann_fun


if __name__ == '__main__':

    v_range = np.arange(-100, 50, 0.1)

    # steady-state: Eder
    vh_act_exp = -34.7
    k_act_exp = -12
    vh_inact_exp = -80.3
    k_inact_exp = 12
    act_exp = boltzmann_fun(v_range, vh_act_exp, k_act_exp)
    inact_exp = boltzmann_fun(v_range, vh_inact_exp, k_inact_exp)

    # Fits
    a = 3.53995631e-01
    b = 1.09610643e+01
    k = -6.02314010e-01
    alpha_fit = rate_constant(v_range, a, b, k)
    a = -3.09741235e-02
    b = -3.91403952e+00
    k = 4.00000000e+01
    beta_fit = rate_constant(v_range, a, b, k)
    act_fit = alpha_fit / (alpha_fit+beta_fit)
    time_constant_act_fit = compute_tau(alpha_fit, beta_fit)

    a = -1.34040177e-01
    b = -4.55898609e+01
    k = 40.0
    alpha_fit = rate_constant(v_range, a, b, k)
    a = 1.00000000e-03
    b = 7.93003639e-02
    k = -40.0
    beta_fit = rate_constant(v_range, a, b, k)

    inact_fit = alpha_fit / (alpha_fit+beta_fit)
    time_constant_inact_fit = compute_tau(alpha_fit, beta_fit)

    pl.figure()
    pl.plot(v_range, act_exp, 'darkred', label='Activation (exp)', linewidth=1.5)
    pl.plot(v_range, act_fit, 'red', label='Activation (fit)', linewidth=1.5)
    pl.plot(v_range, inact_exp, 'darkblue', label='Inactivation (exp)', linewidth=1.5)
    pl.plot(v_range, inact_fit, '#a6bddb', label='Inactivation (fit)', linewidth=1.5)
    pl.plot(v_range, boltzmann_fun(v_range, -71, 30), 'g', label='test', linewidth=1.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('Steady-state curve', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()

    pl.figure()
    pl.plot(v_range, time_constant_act_fit, color='red', label='Activation (fit)', linewidth=1.5)
    pl.plot(v_range, time_constant_inact_fit, color='#a6bddb', label='Inactivation (fit)', linewidth=1.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('Tau (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()