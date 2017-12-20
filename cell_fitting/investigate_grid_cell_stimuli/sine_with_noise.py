from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
import json
import copy
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.evaluation.plot_sine_stimulus import get_sine_stimulus
from bac_project.connectivity.connection import synaptic_input
from cell_fitting.optimization.simulate import iclamp_handling_onset
from time import time
import random


def synaptic_noise_input():
    pos = 0.5
    freq = {'AMPA': 800, 'NMDA': 80, 'GABA': 35000}
    seeds_stim = {'AMPA': time(), 'NMDA': time(), 'GABA': time()}
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


def get_sines(random_generator, field_pos, position, time):
    amp1 = 0.5  # 0.45
    amp2 = 0.2  # 0.12
    freq2 = 8
    sine1_dur_mu = 2000  # ms
    sine1_dur_sig = 400  # ms

    # intervals between fields
    field_pos_t = [time[np.argmin(np.abs(position-p))] for p in field_pos]
    field_intervals_t = np.concatenate((np.array([field_pos_t[0]]), np.diff(field_pos_t),
                                        np.array([time[-1]-field_pos_t[-1]])))

    # draw sine durations
    sine1_durs = draw_sines(random_generator, sine1_dur_mu, sine1_dur_sig, len(field_pos))
    sine1_intervals_dur = np.array([sine1_durs[0]/2]
                                   + [(d1+d2)/2 for d1,d2 in zip(sine1_durs[:-1], sine1_durs[1:])]
                                   + [sine1_durs[-1]/2])
    while ~np.all(sine1_intervals_dur < field_intervals_t):
        sine1_durs = draw_sines(random_generator, sine1_dur_mu, sine1_dur_sig, len(field_pos))
        sine1_intervals_dur = np.array([sine1_durs[0] / 2]
                                       + [(d1 + d2) / 2 for d1, d2 in zip(sine1_durs[:-1], sine1_durs[1:])]
                                       + [sine1_durs[-1] / 2])

    # compute onset times
    onset_durs = field_intervals_t - sine1_intervals_dur

    # compute current traces
    i_sines = [0] * len(sine1_durs)
    for i, (sine1_dur, onset_dur) in enumerate(zip(sine1_durs, onset_durs)):
        if i == len(sine1_durs)-1:
            i_sines[i] = get_sine_stimulus(amp1, amp2, sine1_dur, freq2, onset_dur, onset_durs[i+1], dt)
        else:
            i_sines[i] = get_sine_stimulus(amp1, amp2, sine1_dur, freq2, onset_dur, 0, dt)
    run_stim = np.concatenate(i_sines)
    sine_params = {'amp1': amp1, 'amp2': amp2,
                   'sine1_dur_mu': sine1_dur_mu, 'sine1_dur_sig': sine1_dur_sig,
                   'freq2': freq2}
    return sine_params, run_stim


def draw_sines(random_generator, sine1_dur_mu, sine1_dur_sig, n_sines):
    sine1_durs = np.zeros(n_sines)
    for i in range(n_sines):
        sine1_dur = random_generator.gauss(sine1_dur_mu, sine1_dur_sig)
        while (sine1_dur <= 0):
            sine1_dur = random_generator.gauss(sine1_dur_mu, sine1_dur_sig)
        sine1_durs[i] = sine1_dur
    return sine1_durs


def speed_ar_model(random_generator, dt):
    ar_model = lambda x, a, b, sig: b + a * x + random_generator.gauss(0, sig)
    a = 1
    b = 0
    sig = 0.00001  # ms
    speed_offset = 0.04  # cm/ms
    speed_params = {'speed_type': 'ar', 'a': a, 'b': b, 'sig': sig, 'speed_offset': speed_offset}

    position = [0]
    speed = [0]
    while position[-1] <= track_len:
        new_speed = ar_model(speed[-1], a, b, sig)
        while new_speed + speed_offset <= 0:  # no backward running
            new_speed = ar_model(speed[-1], a, b, sig)
        speed.append(new_speed)
        position.append(position[-1] + (new_speed + speed_offset) * dt)
    position = np.array(position)
    speed = np.array(speed) + speed_offset
    time = np.arange(0, len(position)) * dt
    return speed, position, time, speed_params


if __name__ == '__main__':
    load_mechanism_dir("/home/cf/Phd/programming/projects/bac_project/bac_project/connectivity/vecstim")

    # parameters
    folder = 'test0'
    save_dir = './results/'+folder+'/data'
    #save_dir_model = '../results/server/2017-07-06_13:50:52/434/L-BFGS-B/'
    #model_dir = os.path.join(save_dir_model, 'model', 'cell.json')
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../model/channels/vavoulis'

    onset = 200
    dt = 0.01
    celsius = 35
    v_init = -75
    n_runs = 14
    track_len = 400  # cm
    n_fields = 3
    speed_type = 'constant'
    field_pos = np.cumsum([track_len / n_fields] * n_fields) - (track_len / n_fields) / 2
    seed = time()
    params = {'model_dir': model_dir, 'mechanism_dir': mechanism_dir, 'onset': onset, 'dt': dt, 'celsius': celsius,
              'v_init': v_init, 'n_runs': n_runs, 'n_fields': n_fields, 'track_len': track_len, 'seed': seed}


    # # jiggle field positions
    # field_sig = 1  # cm
    # pos_fields_tmp = pos_fields + np.array([random_generator.gauss(0, field_sig) for i in range(n_fields)])
    # while pos_fields_tmp[0] <= 0 and pos_fields_tmp[-1] >= track_len and np.any(pos_fields_tmp != np.sort(pos_fields_tmp)):
    #     pos_fields_tmp = pos_fields + np.array([random_generator.gauss(0, field_sig) for i in range(n_fields)])
    # pos_fields = pos_fields_tmp

    # random generator
    random_generator = random.Random()
    random_generator.seed(seed)

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # simulate animal position
    positions = [0] * n_runs
    speeds = [0] * n_runs
    times = [0] * n_runs
    for i_run in range(n_runs):
        if speed_type == 'constant':
            lower_bound = 0.01
            upper_bound = 0.07
            speed = random_generator.uniform(lower_bound, upper_bound)
            positions[i_run] = np.arange(0, track_len+speed*dt, speed*dt)
            speeds[i_run] = np.ones(len(positions[i_run])) * speed
            times[i_run] = np.arange(0, len(positions[i_run])) * dt
            speed_params = {'speed_type': speed_type, 'lower_bound': lower_bound, 'upper_bound': upper_bound}
        elif speed_type == 'ar':
            speeds[i_run], positions[i_run], times[i_run], speed_params = speed_ar_model(random_generator, dt)

    # input
    sine_stims = [0] * n_runs
    for i_run in range(n_runs):
        sine_params, sine_stims[i_run] = get_sines(random_generator, field_pos, positions[i_run], times[i_run])
    sine_stimulus = np.concatenate(sine_stims)
    tstop = (len(np.concatenate(positions))-1) * dt
    syn_params, AMPA_stimulation, NMDA_stimulation, GABA_stimulation = synaptic_noise_input()

    # simulate
    simulation_params = {'sec': ('soma', None), 'i_inj': sine_stimulus, 'v_init': v_init, 'tstop': tstop,
                         'dt': dt, 'celsius': celsius, 'onset': onset}
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'v.npy'), v)
    np.save(os.path.join(save_dir, 't.npy'), t)
    np.save(os.path.join(save_dir, 'position.npy'), np.concatenate(positions))
    np.save(os.path.join(save_dir, 'speed.npy'), np.concatenate(speeds))

    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(params, f)
    with open(os.path.join(save_dir, 'syn_params.json'), 'w') as f:
        json.dump(syn_params, f)
    with open(os.path.join(save_dir, 'sine_params.json'), 'w') as f:
        json.dump(sine_params, f)
    with open(os.path.join(save_dir, 'speed_params.json'), 'w') as f:
        json.dump(sine_params, f)

    pl.figure()
    pl.plot(t, v, 'k')
    pl.ylabel('Membrane potential (mV)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.savefig(os.path.join(save_dir, 'v.svg'))
    pl.show()

    pl.figure()
    pl.plot(t, np.concatenate(positions), 'k')
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Position (cm)', fontsize=16)
    pl.savefig(os.path.join(save_dir, 'position.svg'))
    pl.show()

    # TODO: define sine variation as function of space