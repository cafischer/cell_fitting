import pylab as pl
import numpy as np
import json
from matplotlib.pyplot import cm
from new_optimization.fitter import iclamp_handling_onset, FitterFactory
from new_optimization.evaluation.evaluate import get_best_candidate, get_candidate_params

__author__ = 'caro'


def get_ramp(start_idx, end_idx, amp_before, ramp_amp, amp_after):
    diff_idx = end_idx - start_idx
    half_diff_up = int(round(diff_idx / 2)) - 1
    half_diff_down = int(round(diff_idx / 2)) + 1  # peak is one earlier
    if diff_idx % 2 != 0:
        half_diff_down += 1
    i_exp = np.zeros(diff_idx)
    i_exp[:half_diff_up] = np.linspace(amp_before, ramp_amp, half_diff_up)
    i_exp[half_diff_up:] = np.linspace(ramp_amp, amp_after, half_diff_down+1)[1:]
    return i_exp


def hyperpolarize_ramp(cell):

    hyperamps = np.arange(-0.25, 0.26, 0.05)  # nA
    ramp_amp = 8  # nA
    dt = 0.01
    hyp_st_ms = 200  # ms
    hyp_end_ms = 600  # ms
    ramp_end_ms = 602  # ms
    tstop = 1000  # ms

    hyp_st = int(round(hyp_st_ms / dt))
    hyp_end = int(round(hyp_end_ms / dt))
    ramp_end = int(round(ramp_end_ms / dt)) + 1

    t_exp = np.arange(0, tstop + dt, dt)

    v = np.zeros([len(hyperamps), len(t_exp)])
    for j, hyper_amp in enumerate(hyperamps):
        i_exp = np.zeros(len(t_exp))
        i_exp[hyp_st:hyp_end] = hyper_amp
        i_exp[hyp_end:ramp_end] = get_ramp(hyp_end, ramp_end, hyper_amp, ramp_amp, 0)

        # get simulation parameters
        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -59, 'tstop': t_exp[-1],
                             'dt': dt, 'celsius': 35, 'onset': 200}

        # record v
        v[j], t, _ = iclamp_handling_onset(cell, **simulation_params)

        # test i_exp
        #pl.figure()
        #pl.plot(t, i_exp)
        #pl.show()

    # plot
    pl.figure()
    color = iter(cm.gist_rainbow(np.linspace(0, 1, len(hyperamps))))
    for j, hyper_amp in enumerate(hyperamps):
        pl.plot(t, v[j], c=next(color), label=str(np.round(hyper_amp, 2)) + ' nA')
    pl.xlabel('Time $(ms)$', fontsize=16)
    pl.ylabel('Membrane potential $(mV)$', fontsize=16)
    pl.legend(loc='upper right', fontsize=16)
    pl.show()


if __name__ == '__main__':
    # parameters
    #save_dir = '../../results/new_optimization/2015_08_06d/15_02_17_PP(4)/L-BFGS-B/'
    save_dir = '../../results/server/2017-06-19_13:12:49/189/L-BFGS-B'

    # load model
    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    best_candidate = get_candidate_params(get_best_candidate(save_dir, n_best=0))
    fitter.update_cell(best_candidate)
    hyperpolarize_ramp(fitter.cell)