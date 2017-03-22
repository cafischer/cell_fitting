import numpy as np
import matplotlib.pyplot as pl
from test_channels.channel_characteristics import rate_constant, compute_tau, boltzmann_fun


if __name__ == '__main__':

    v_range = np.arange(-100, 50, 0.1)

    # steady-state: Eder
    vh_act_exp = -4.8
    k_act_exp = -11
    vh_inact_exp = -74.3
    k_inact_exp = 15
    act_exp = boltzmann_fun(v_range, vh_act_exp, k_act_exp)
    inact_exp = boltzmann_fun(v_range, vh_inact_exp, k_inact_exp)
    inact_exp = inact_exp / 2 + 0.5

    # Fits
    a = 2.87088805e-02
    b = -1.37069649
    k = -1.38932639e+01
    alpha_fit = rate_constant(v_range, a, b, k)
    a = -1.20587039e-01
    b = -3.19538122e+01
    k = 40.0
    beta_fit = rate_constant(v_range, a, b, k)
    act_fit = alpha_fit / (alpha_fit+beta_fit)
    time_constant_act_fit = compute_tau(alpha_fit, beta_fit)

    a = -1.00000000e-03
    b = -5.28299423e+01
    k = 1.51287390e+01
    alpha_fit = rate_constant(v_range, a, b, k)
    a = 1.15851564e-01
    b = -5.19260809e+01
    k = -40.0
    beta_fit = rate_constant(v_range, a, b, k)

    inact_fit = alpha_fit / (alpha_fit+beta_fit)
    time_constant_inact_fit = compute_tau(alpha_fit, beta_fit)

    pl.figure()
    pl.plot(v_range, act_exp, 'darkred', label='Activation (exp)', linewidth=1.5)
    pl.plot(v_range, act_fit, 'red', label='Activation (fit)', linewidth=1.5)
    pl.plot(v_range, inact_exp, 'darkblue', label='Inactivation (exp)', linewidth=1.5)
    pl.plot(v_range, inact_fit, '#a6bddb', label='Inactivation (fit)', linewidth=1.5)
    #pl.plot(v_range, boltzmann_fun(v_range, -1, -13), 'g', label='test', linewidth=1.5)
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