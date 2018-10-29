from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
import copy
from nrn_wrapper import Cell, load_mechanism_dir
from bac_project.connectivity.connection import synaptic_input
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_characteristics import to_idx
from time import time
from statsmodels import api as sm


def synaptic_noise_input():
    pos = 0.5
    #freq = {'AMPA': 10000, 'NMDA': 200, 'GABA': 30000}
    freq = {'AMPA': 20000, 'NMDA': 0, 'GABA': 20000}
    seeds_stim = {'AMPA': time(), 'NMDA': time(), 'GABA': time()}
    n_stim = {'AMPA': 1, 'NMDA': 1, 'GABA': 1}
    params_stim = {'kind': 'poisson', 'onset': onset, 'tstop': tstop, 'dt': dt,
                   'freq': freq['AMPA'], 'pos': pos, 'seed': seeds_stim['AMPA']}
    params_syn = {'tau1': 0.25, 'tau2': 2.5, 'e': 0, 'pos': pos}
    params_weight = {'kind': 'same', 'weight': 0.0001}
    params_AMPA = {'cell': cell, 'n_stim': n_stim['AMPA'], 'section': 'soma', 'params_stim': params_stim,
                   'params_syn': params_syn, 'params_weight': params_weight, 'delay': 0}
    params_stim = {'kind': 'poisson', 'onset': onset, 'tstop': tstop, 'dt': dt,
                   'freq': freq['NMDA'], 'pos': pos, 'seed': seeds_stim['NMDA']}
    params_syn = {'tau1': 5, 'tau2': 150, 'e': 0, 'pos': pos}
    params_weight = {'kind': 'same', 'weight': 0.0001}
    params_NMDA = {'cell': cell, 'n_stim': n_stim['NMDA'], 'section': 'soma', 'params_stim': params_stim,
                   'params_syn': params_syn, 'params_weight': params_weight, 'delay': 0}
    params_stim = {'kind': 'poisson', 'onset': onset, 'tstop': tstop, 'dt': dt,
                   'freq': freq['GABA'], 'pos': pos, 'seed': seeds_stim['GABA']}
    params_syn = {'tau1': 0.5, 'tau2': 5, 'e': -75, 'pos': pos}
    params_weight = {'kind': 'same', 'weight': 0.0001}
    params_GABA = {'cell': cell, 'n_stim': n_stim['GABA'], 'section': 'soma', 'params_stim': params_stim,
                   'params_syn': params_syn, 'params_weight': params_weight, 'delay': 0}

    params_AMPA_s = copy.deepcopy(params_AMPA)
    params_AMPA_s['cell'] = None
    params_NMDA_s = copy.deepcopy(params_NMDA)
    params_NMDA_s['cell'] = None
    params_GABA_s = copy.deepcopy(params_GABA)
    params_GABA_s['cell'] = None
    syn_params = [params_AMPA_s, params_NMDA_s, params_GABA_s]

    syn_AMPA, stim_AMPA, con_AMPA, weights_AMPA, spiketimes_AMPA = synaptic_input(**params_AMPA)
    syn_NMDA, stim_NMDA, con_NMDA, weights_NMDA, spiketimes_NMDA = synaptic_input(**params_NMDA)
    syn_GABA, stim_GABA, con_GABA, weights_GABA, spiketimes_GABA = synaptic_input(**params_GABA)
    AMPA_stimulation = [syn_AMPA, stim_AMPA, con_AMPA, weights_AMPA, spiketimes_AMPA]
    NMDA_stimulation = [syn_NMDA, stim_NMDA, con_NMDA, weights_NMDA, spiketimes_NMDA]
    GABA_stimulation = [syn_GABA, stim_GABA, con_GABA, weights_GABA, spiketimes_GABA]

    return syn_params, AMPA_stimulation, NMDA_stimulation, GABA_stimulation


if __name__ == '__main__':
    load_mechanism_dir("/home/cf/Phd/programming/projects/bac_project/bac_project/connectivity/vecstim")

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    load_mechanism_dir(mechanism_dir)

    onset = 200
    dt = 0.01
    celsius = 35
    v_init = -75
    tstop = 1000
    n_trials = 10

    # create cell
    cell = Cell.from_modeldir(model_dir)

    # simulate
    medians = np.zeros(n_trials)
    means = np.zeros(n_trials)
    stds = np.zeros(n_trials)
    v_traces = []
    t_traces = []
    for n in range(n_trials):
        syn_params, AMPA_stimulation, NMDA_stimulation, GABA_stimulation = synaptic_noise_input()
        simulation_params = {'sec': ('soma', None), 'i_inj': np.zeros(to_idx(tstop, dt)), 'v_init': v_init, 'tstop': tstop,
                             'dt': dt, 'celsius': celsius, 'onset': onset}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)
        medians[n] = np.median(v)
        means[n] = np.mean(v)
        stds[n] = np.std(v)
        v_traces.append(v)
        t_traces.append(t)

        # # plot
        # pl.figure()
        # pl.plot(t, v, 'k')
        # pl.ylabel('Membrane potential (mV)', fontsize=16)
        # pl.xlabel('Time (ms)', fontsize=16)
        # pl.tight_layout()
        #
        # pl.figure()
        # pl.hist(v, bins=100)
        # pl.ylabel('Count', fontsize=16)
        # pl.xlabel('Membrane potential (mV)', fontsize=16)
        # pl.tight_layout()
        # pl.show()

    print 'Average Median: %.2f' % np.mean(medians)
    print 'Average Mean: %.2f' % np.mean(means)
    print 'Average Std: %.2f' % np.mean(stds)

    for v, t in zip(v_traces, t_traces):
        auto_corr = sm.tsa.acf(v, nlags=len(v))
        pl.figure()
        pl.plot(t, auto_corr)
        pl.xlabel('Time lag (ms)')
        pl.ylabel('Autocorrelation')
        pl.show()