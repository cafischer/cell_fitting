import numpy as np
import matplotlib.pyplot as pl
from test_channels.channel_characteristics import rate_constant, compute_tau, boltzmann_fun


if __name__ == '__main__':

    v_range = np.arange(-100, 50, 0.1)

    # steady-state: Magistretti
    vh_act_exp = -32.5
    k_act_exp = -3.6
    vh_inact_exp = -59.8
    k_inact_exp = 4.5
    act_exp = boltzmann_fun(v_range, vh_act_exp, k_act_exp)
    inact_exp = boltzmann_fun(v_range, vh_inact_exp, k_inact_exp)

    # Fits
    a = 0.20342639
    b = -19.05427426
    k = -40.0
    alpha_fit = rate_constant(v_range, a, b, k)
    a = -0.46482694
    b = -12.60876073
    k = 4.31723911
    beta_fit = rate_constant(v_range, a, b, k)
    act_fit = alpha_fit / (alpha_fit+beta_fit)
    time_constant_act_fit = compute_tau(alpha_fit, beta_fit)

    a = -0.26247431
    b = -96.12589008
    k = 40.0
    alpha_fit = rate_constant(v_range, a, b, k)
    a = 2.0
    b = 30.50199261
    k = -3.68528387
    beta_fit = rate_constant(v_range, a, b, k)

    inact_fit = alpha_fit / (alpha_fit+beta_fit)
    time_constant_inact_fit = compute_tau(alpha_fit, beta_fit)

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
    pl.plot(v_range, time_constant_act_fit, color='red', label='Activation (fit)', linewidth=1.5)
    pl.plot(v_range, time_constant_inact_fit, color='#a6bddb', label='Inactivation (fit)', linewidth=1.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('Tau (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()