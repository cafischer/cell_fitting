import numpy as np
import matplotlib.pyplot as pl
from test_channels.channel_characteristics import rate_constant, compute_tau, boltzmann_fun
import os
import json


if __name__ == '__main__':

    save_dir = './plots/fit_vsteps'
    data_dir = '/media/caro/Daten/Phd/DAP-Project/cell_fitting/results/ion_channels/hcn/L-BFGS-B/best_candidate.json'
    v_range = np.arange(-100, 50, 0.1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # steady-state: Dickson
    #vh_fast_exp = -67.4
    #k_fast_exp = 12.66
    #vh_slow_exp = -57.92
    #k_slow_exp = 9.26
    #steadystate_fast_exp = boltzmann_fun(v_range, vh_fast_exp, k_fast_exp)
    #steadystate_slow_exp = boltzmann_fun(v_range, vh_slow_exp, k_slow_exp)

    # time cosntant: Dickson
    a = -2.89 * 1e-3
    b = -0.445
    k = 24.02
    alpha_exp = rate_constant(v_range, a, b, k)
    a = 2.71 * 1e-2
    b = -1.024
    k = -17.4
    beta_exp = rate_constant(v_range, a, b, k)
    time_constant_fast_exp = compute_tau(alpha_exp, beta_exp)
    steadystate_fast_exp = alpha_exp / (alpha_exp+beta_exp)

    a = -3.18 * 1e-3
    b = -0.695
    k = 26.72
    alpha_exp = rate_constant(v_range, a, b, k)
    a = 2.16 * 1e-2
    b = -1.065
    k = -14.25
    beta_exp = rate_constant(v_range, a, b, k)
    time_constant_slow_exp = compute_tau(alpha_exp, beta_exp)
    steadystate_slow_exp = alpha_exp / (alpha_exp+beta_exp)

    # Fits
    with open(data_dir, 'r') as f:
        best_candidate = json.load(f)

    a = best_candidate['a_alpha_m']
    b = best_candidate['b_alpha_m']
    k = best_candidate['k_alpha_m']
    alpha_fit = rate_constant(v_range, a, b, k)
    a = best_candidate['a_beta_m']
    b = best_candidate['b_beta_m']
    k = best_candidate['k_beta_m']
    beta_fit = rate_constant(v_range, a, b, k)
    time_constant_fast_fit = compute_tau(alpha_fit, beta_fit)
    steadystate_fast_fit = alpha_fit / (alpha_fit + beta_fit)

    a = best_candidate['a_alpha_h']
    b = best_candidate['b_alpha_h']
    k = best_candidate['k_alpha_h']
    alpha_fit = rate_constant(v_range, a, b, k)
    a = best_candidate['a_beta_h']
    b = best_candidate['b_beta_h']
    k = best_candidate['k_beta_h']
    beta_fit = rate_constant(v_range, a, b, k)
    time_constant_slow_fit = compute_tau(alpha_fit, beta_fit)
    steadystate_slow_fit = alpha_fit / (alpha_fit + beta_fit)

    pl.figure()
    pl.plot(v_range, steadystate_fast_exp, 'darkred', label='Fast activation (exp)', linewidth=1.5)
    pl.plot(v_range, steadystate_fast_fit, 'red', label='Fast activation (fit)', linewidth=1.5)
    pl.plot(v_range, steadystate_slow_exp, 'darkblue', label='Slow activation (exp)', linewidth=1.5)
    pl.plot(v_range, steadystate_slow_fit, '#a6bddb', label='Slow activation (fit)', linewidth=1.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('Steady-state curve', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'steadystate_comparison.png'))
    pl.show()

    pl.figure()
    pl.plot(v_range, time_constant_fast_exp, color='darkred', label='Fast activation (exp)', linewidth=1.5)
    pl.plot(v_range, time_constant_slow_exp, color='darkblue', label='Slow activation (exp)', linewidth=1.5)
    pl.plot(v_range, time_constant_fast_fit, color='red', label='Fast activation (fit)', linewidth=1.5)
    pl.plot(v_range, time_constant_slow_fit, color='#a6bddb', label='Slow activation (fit)', linewidth=1.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('Tau (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'timeconstant_comparison.png'))
    pl.show()