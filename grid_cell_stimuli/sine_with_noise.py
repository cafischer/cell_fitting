from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
from new_optimization.evaluation.sine_stimulus import get_sine_stimulus
from bac_project.connectivity.connection import synaptic_input
from optimization.simulate import iclamp_handling_onset


def synaptic_noise_input():
    pos = 0.5
    freq = {'AMPA': 800, 'NMDA': 80, 'GABA': 35000}
    seeds_stim = {'AMPA': 1, 'NMDA': 1, 'GABA': 1}
    n_stim = {'AMPA': 1, 'NMDA': 1, 'GABA': 1}
    params_stim = {'kind': 'poisson', 'onset': onset, 'tstop': tstop, 'dt': dt,
                   'freq': freq['AMPA'], 'pos': pos, 'seed': seeds_stim['AMPA']}
    params_syn = {'tau1': 0.25, 'tau2': 2.5, 'e': 0, 'pos': pos}
    params_weight = {'kind': 'same', 'weight': 0.001}
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

    syn_AMPA, stim_AMPA, con_AMPA, weights_AMPA, spiketimes_AMPA = synaptic_input(**params_AMPA)
    syn_NMDA, stim_NMDA, con_NMDA, weights_NMDA, spiketimes_NMDA = synaptic_input(**params_NMDA)
    syn_GABA, stim_GABA, con_GABA, weights_GABA, spiketimes_GABA = synaptic_input(**params_GABA)
    AMPA_stimulation = [syn_AMPA, stim_AMPA, con_AMPA, weights_AMPA, spiketimes_AMPA]
    NMDA_stimulation = [syn_NMDA, stim_NMDA, con_NMDA, weights_NMDA, spiketimes_NMDA]
    GABA_stimulation = [syn_GABA, stim_GABA, con_GABA, weights_GABA, spiketimes_GABA]
    return AMPA_stimulation, NMDA_stimulation, GABA_stimulation


def sine_input():
    amp1 = 0.45
    amp2 = 0.12
    sine1_dur = 1000
    freq2 = 8
    onset_dur = 3000
    offset_dur = 3000
    tstop = sine1_dur + onset_dur + offset_dur
    i_sine = get_sine_stimulus(amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)
    return i_sine, tstop


if __name__ == '__main__':
    load_mechanism_dir("/home/cf/Phd/programming/projects/bac_project/bac_project/connectivity/vecstim")

    # parameters
    save_dir = './results/test0/data'
    save_dir_model = '../results/server/2017-07-06_13:50:52/434/L-BFGS-B/'
    model_dir = os.path.join(save_dir_model, 'model', 'cell.json')
    mechanism_dir = '../model/channels/vavoulis'

    onset = 200
    dt = 0.01
    celsius = 35
    v_init = -75

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # input
    i_sine, tstop = sine_input()
    AMPA_stimulation, NMDA_stimulation, GABA_stimulation = synaptic_noise_input()

    # simulate
    simulation_params = {'sec': ('soma', None), 'i_inj': i_sine, 'v_init': v_init, 'tstop': tstop,
                         'dt': dt, 'celsius': celsius, 'onset': onset}
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pl.figure()
    pl.plot(t, v, 'k')
    pl.ylabel('Membrane potential (mV)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.savefig(os.path.join(save_dir, 'v.png'))
    pl.show()

    np.save(os.path.join(save_dir, 'v.npy'), v)
    np.save(os.path.join(save_dir, 't.npy'), t)