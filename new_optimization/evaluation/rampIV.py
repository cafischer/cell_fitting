import pylab as pl
import numpy as np
import json
from new_optimization.fitter import iclamp_handling_onset, FitterFactory
from new_optimization.evaluation.evaluate import get_best_candidate, get_candidate_params

__author__ = 'caro'


def get_ramp(start_idx, peak_idx, end_idx, amp_before, ramp_amp, amp_after):
    diff_idx = end_idx - start_idx
    half_diff_up = peak_idx - start_idx + 1
    half_diff_down = end_idx - peak_idx - 1
    i_exp = np.zeros(diff_idx)
    i_exp[:half_diff_up] = np.linspace(amp_before, ramp_amp, half_diff_up)
    i_exp[half_diff_up:] = np.linspace(ramp_amp, amp_after, half_diff_down+1)[1:]
    return i_exp


def rampIV(cell, ramp_amp, v_init=-75):

    dt = 0.01
    ramp_st_ms = 10  # ms
    ramp_peak_ms = 10.8  # ms
    ramp_end_ms = 12  # ms
    tstop = 161.99  # ms

    ramp_st = int(round(ramp_st_ms / dt))
    ramp_peak = int(round(ramp_peak_ms / dt))
    ramp_end = int(round(ramp_end_ms / dt)) + 1

    t_exp = np.arange(0, tstop + dt, dt)
    i_exp = np.zeros(len(t_exp))
    i_exp[ramp_st:ramp_end] = get_ramp(ramp_st, ramp_peak, ramp_end, 0, ramp_amp, 0)

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init, 'tstop': t_exp[-1],
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    # test i_exp
    #pl.figure()
    #pl.plot(t, i_exp)
    #pl.show()

    # plot
    pl.figure()
    pl.plot(t, v, 'r', label=str(np.round(ramp_amp, 2)) + ' nA')
    pl.xlabel('Time $(ms)$', fontsize=16)
    pl.ylabel('Membrane potential $(mV)$', fontsize=16)
    pl.legend(loc='upper right', fontsize=16)
    pl.show()


if __name__ == '__main__':
    # parameters
    #save_dir = '../../results/new_optimization/2015_08_06d/15_02_17_PP(4)/L-BFGS-B/'
    save_dir = '../../results/server/2017-06-19_13:12:49/189/L-BFGS-B'
    ramp_amp = 2.8

    # load model
    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    best_candidate = get_candidate_params(get_best_candidate(save_dir, n_best=0))
    fitter.update_cell(best_candidate)
    rampIV(fitter.cell, ramp_amp, v_init=-75)