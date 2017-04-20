import pylab as pl
import numpy as np
import json
from matplotlib.pyplot import cm
from new_optimization.fitter import iclamp_handling_onset, FitterFactory
from new_optimization.evaluation.evaluate import get_best_candidate, get_candidate_params

__author__ = 'caro'


def hyperpolarize_ramp(cell):

    hyperamps = np.arange(-0.9, 0.11, 0.2)
    rampamp = 3.0
    dt = 0.01
    hyp_st_ms = 4000.0 * dt
    hyp_end_ms = 16000.0 * dt  # 12000
    ramp_end_ms = 16400.0 * dt  # 12040.0*dt

    t_exp = np.arange(0, hyp_st_ms + hyp_end_ms + ramp_end_ms, dt)

    v = np.zeros([len(hyperamps), len(t_exp)])
    for j, hyperamp in enumerate(hyperamps):
        i_exp = np.zeros(len(t_exp))
        i_exp[int(round(hyp_st_ms / dt, 0)):int(round(hyp_end_ms / dt, 0))] = hyperamp
        i_exp[int(round(hyp_end_ms / dt, 0)):int(round(hyp_end_ms / dt + (ramp_end_ms - hyp_end_ms) / dt / 2, 0))] = \
            np.linspace(hyperamp, rampamp, len(i_exp[int(round(hyp_end_ms / dt, 0)):
            int(round(hyp_end_ms / dt + (ramp_end_ms - hyp_end_ms) / dt / 2, 0))]))
        i_exp[int(round(hyp_end_ms / dt + (ramp_end_ms - hyp_end_ms) / dt / 2, 0)):int(round(hyp_end_ms / dt, 0))] = \
            np.linspace(rampamp, 0.0, len(i_exp[int(round(hyp_end_ms / dt + (ramp_end_ms - hyp_end_ms) / dt / 2, 0)):
            int(round(hyp_end_ms / dt, 0))]))

        # get simulation parameters
        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -59, 'tstop': t_exp[-1],
                             'dt': dt, 'celsius': 35, 'onset': 200}

        # record v
        v[j], t, _ = iclamp_handling_onset(cell, **simulation_params)

    # plot
    pl.figure()
    color = iter(cm.gist_rainbow(np.linspace(0, 1, len(hyperamps))))
    for j, hyperamp in enumerate(hyperamps):
        pl.plot(t, v[j], c=next(color), label='amp: ' + str(hyperamp))
    pl.xlabel('Time $(ms)$', fontsize=16)
    pl.ylabel('Membrane potential $(mV)$', fontsize=16)
    pl.legend(loc='upper right', fontsize=16)
    pl.show()


if __name__ == '__main__':
    # parameters
    #save_dir = '../../results/new_optimization/2015_08_06d/15_02_17_PP(4)/L-BFGS-B/'
    save_dir = '../../results/server/2017-04-11_20:44:13/34/L-BFGS-B/'

    # load model
    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    best_candidate = get_candidate_params(get_best_candidate(save_dir, n_best=0))
    fitter.update_cell(best_candidate)
    hyperpolarize_ramp(fitter.cell)

