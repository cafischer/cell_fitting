from __future__ import division
import pylab as pl
import numpy as np
import json
from matplotlib.pyplot import cm
from new_optimization.evaluation.evaluate import FitterFactory, get_best_candidate, get_candidate_params
from optimization.simulate import iclamp_handling_onset

__author__ = 'caro'


def get_ramp(start_idx, end_idx, amp_before, ramp_amp, amp_after):
    diff_idx = end_idx - start_idx
    half_diff_up = int(np.ceil(diff_idx / 2))
    half_diff_down = int(np.floor(diff_idx / 2))
    i_exp = np.zeros(diff_idx)
    i_exp[:half_diff_up] = np.linspace(amp_before, ramp_amp, half_diff_up)
    i_exp[half_diff_up:] = np.linspace(ramp_amp, amp_after, half_diff_down+1)[1:]
    return i_exp


def double_ramp(cell):
    """
    original values
    delta_ramp = 2
    delta_first = 3
    ramp3_times = np.arange(delta_first, 12 * delta_ramp + delta_ramp, delta_ramp)
    baseline_amp = -0.05
    ramp_amp = 4.0
    ramp3_amp = 1.8
    step_amp = 0  # or -0.1 or 0.1
    dt = 0.01
    """

    delta_ramp = 2
    delta_first = 3
    ramp3_times = np.arange(delta_first, 12 * delta_ramp + delta_ramp, delta_ramp)
    baseline_amp = 0.0 #-0.05
    ramp_amp = 4.0
    ramp3_amp = 1.8
    step_amp = -0.1
    dt = 0.01

    # construct current traces
    len_ramp = 3
    start_ramp1 = int(round(20 / dt))
    end_ramp1 = start_ramp1 + int(round(len_ramp / dt))
    start_step = int(round(222 / dt))
    end_step = start_step + int(round(250 / dt))
    start_ramp2 = end_step + int(round(15 / dt))
    end_ramp2 = start_ramp2 + int(round(len_ramp / dt))

    t_exp = np.arange(0, 800, dt)
    v = np.zeros([len(ramp3_times), len(t_exp)])

    for j, ramp3_time in enumerate(ramp3_times):
        start_ramp3 = end_ramp2 + int(round(ramp3_time / dt))
        end_ramp3 = start_ramp3 + int(round(len_ramp / dt))

        i_exp = np.ones(len(t_exp)) * baseline_amp
        i_exp[start_ramp1:end_ramp1] = get_ramp(start_ramp1, end_ramp1, 0, ramp_amp, 0)
        i_exp[start_step:end_step] = step_amp
        i_exp[start_ramp2:end_ramp2] = get_ramp(start_ramp2, end_ramp2, 0, ramp_amp, 0)
        i_exp[start_ramp3:end_ramp3] = get_ramp(start_ramp3, end_ramp3, 0, ramp3_amp, 0)

        # get simulation parameters
        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -80, 'tstop': t_exp[-1],
                             'dt': dt, 'celsius': 35, 'onset': 300}

        # record v
        v[j], t, _ = iclamp_handling_onset(cell, **simulation_params)

    # plot
    pl.figure()
    color = iter(cm.gist_rainbow(np.linspace(0, 1, len(ramp3_times))))
    for j, ramp3_time in enumerate(ramp3_times):
        pl.plot(t, v[j], label='Model' if j==0 else '', c='r')  # label='time: '+str(ramp3_time), c=next(color))
    #pl.plot(t, v[-1])
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Membrane potential (mV)', fontsize=16)
    #pl.legend(loc='upper right', fontsize=16)
    pl.xlim(485, 560)
    pl.tight_layout()
    pl.legend(fontsize=16)
    pl.show()


if __name__ == '__main__':
    # parameters
    #save_dir = '../../results/optimization_vavoulis_channels/2015_08_06d/10_04_17_readjust/L-BFGS-B/'
    save_dir = '../../results/optimization_vavoulis_channels/2015_08_26b/22_01_17_readjust1/L-BFGS-B/'
    #save_dir = '../../results/server/2017-05-22_10:47:38/371/L-BFGS-B'
    n_best = 0

    # load model
    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    best_candidate = get_candidate_params(get_best_candidate(save_dir, n_best))
    fitter.update_cell(best_candidate)

    double_ramp(fitter.cell)