import numpy as np
import matplotlib.pyplot as pl
from test_channels.channel_characteristics import rate_constant, compute_tau, boltzmann_fun


if __name__ == '__main__':

    v_range = np.arange(-100, 50, 0.1)

    # steady-state: Magistretti
    vh_act_exp = -44.4
    k_act_exp = -5.2
    vh_inact_exp = -48.8
    k_inact_exp = 10
    act_exp = boltzmann_fun(v_range, vh_act_exp, k_act_exp)
    inact_exp = boltzmann_fun(v_range, vh_inact_exp, k_inact_exp)

    # Fits
    vh_act_fit = -30
    k_act_fit = -4.742
    vh_inact_fit = -53
    k_inact_fit = 8
    act_fit = boltzmann_fun(v_range, vh_act_fit, k_act_fit)
    inact_fit = boltzmann_fun(v_range, vh_inact_fit, k_inact_fit)

    a = -0.0001
    b = 0.00113
    k = 5.651
    alpha_fit = rate_constant(v_range, a, b, k)
    a = 0.0997
    b = -0.224
    k = -8.293
    beta_fit = rate_constant(v_range, a, b, k)
    time_constant_inact_fit = compute_tau(alpha_fit, beta_fit)

    # time constants: Magistretti
    a = -2.88 * 1e-3
    b = -4.9 * 1e-2
    k = 4.63
    alpha_exp = rate_constant(v_range, a, b, k)
    a = 6.94 * 1e-3
    b = 0.447
    k = -2.63
    beta_exp = rate_constant(v_range, a, b, k)
    time_constant_inact_exp = compute_tau(alpha_exp, beta_exp)

    pl.figure()
    pl.plot(v_range, act_exp, 'darkred', label='Activation (exp)', linewidth=1.5)
    pl.plot(v_range, act_fit, 'red', label='Activation (fit)', linewidth=1.5)
    pl.plot(v_range, inact_exp, 'darkblue', label='Inactivation (exp)', linewidth=1.5)
    pl.plot(v_range, inact_fit, '#a6bddb', label='Inactivation (fit)', linewidth=1.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('Steady-state curve', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()

    pl.figure()
    pl.plot(v_range, time_constant_inact_exp*1000, color='darkblue', label='Inactivation (exp)', linewidth=1.5)
    pl.plot(v_range, time_constant_inact_fit, color='#a6bddb', label='Inactivation (fit)', linewidth=1.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('Tau (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()