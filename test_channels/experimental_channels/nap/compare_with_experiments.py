import numpy as np
import matplotlib.pyplot as pl
from test_channels.channel_characteristics import rate_constant, compute_tau, boltzmann_fun
import json
import os


if __name__ == '__main__':

    save_dir = './plots/fit_vsteps_linear'
    data_dir = '/media/caro/Daten/Phd/DAP-Project/cell_fitting/results/ion_channels/nap_linear/L-BFGS-B/best_candidate.json'
    v_range = np.arange(-100, 50, 0.1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # steady-state: Magistretti
    vh_act_exp = -44.4
    k_act_exp = -5.2
    vh_inact_exp = -48.8
    k_inact_exp = 10
    steadystate_act_exp = boltzmann_fun(v_range, vh_act_exp, k_act_exp)
    steadystate_inact_exp = boltzmann_fun(v_range, vh_inact_exp, k_inact_exp)

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

    # Fits
    with open(data_dir, 'r') as f:
        best_candidate = json.load(f)
    vh_act_fit = best_candidate['vh_m']
    k_act_fit = best_candidate['k_m']
    steadystate_act_fit = boltzmann_fun(v_range, vh_act_fit, k_act_fit)

    a = best_candidate['a_alpha_h']
    b = best_candidate['b_alpha_h']
    k = best_candidate['k_alpha_h']
    alpha_fit = rate_constant(v_range, a, b, k)
    a = best_candidate['a_beta_h']
    b = best_candidate['b_beta_h']
    k = best_candidate['k_beta_h']
    beta_fit = rate_constant(v_range, a, b, k)
    time_constant_inact_fit = compute_tau(alpha_fit, beta_fit)
    steadystate_inact_fit = alpha_fit / (alpha_fit + beta_fit)

    pl.figure()
    pl.plot(v_range, steadystate_act_exp, 'darkred', label='Activation (exp)', linewidth=1.5)
    pl.plot(v_range, steadystate_act_fit, 'red', label='Activation (fit)', linewidth=1.5)
    pl.plot(v_range, steadystate_inact_exp, 'darkblue', label='Inactivation (exp)', linewidth=1.5)
    pl.plot(v_range, steadystate_inact_fit, '#a6bddb', label='Inactivation (fit)', linewidth=1.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('Steady-state curve', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'steadystate_comparison.png'))
    pl.show()

    pl.figure()
    pl.plot(v_range, time_constant_inact_exp*1000, color='darkblue', label='Inactivation (exp)', linewidth=1.5)
    pl.plot(v_range, time_constant_inact_fit, color='#a6bddb', label='Inactivation (fit)', linewidth=1.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('Tau (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'timeconstant_comparison.png'))
    pl.show()