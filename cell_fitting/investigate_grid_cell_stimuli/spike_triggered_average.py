from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.simulate import iclamp_handling_onset
from time import time
import random
from model_noise.with_OU import ou_noise_input
from cell_fitting.investigate_grid_cell_stimuli.sine_with_noise import get_sines
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_characteristics import to_idx


if __name__ == '__main__':
    # parameters
    model_ids = range(1, 7)
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '../model/channels/vavoulis'
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")
    load_mechanism_dir(mechanism_dir)

    noise_params = {'g_e0': 0.005, 'g_i0': 0.05, 'std_e': 0.007, 'std_i': 0.006, 'tau_e': 2.4, 'tau_i': 5.0}

    before_AP = 20
    after_AP = 40

    onset = 200
    dt = 0.01
    celsius = 35
    v_init = -75
    n_runs = 1
    track_len = 3000  # cm
    n_fields = 2
    speed_type = 'constant'
    field_pos = np.cumsum([track_len / n_fields] * n_fields) - (track_len / n_fields) / 2
    seed = time()

    for model_id in model_ids:

        # random generator
        random_generator = random.Random()
        random_generator.seed(seed)

        # create cell
        cell = Cell.from_modeldir(os.path.join(save_dir, str(model_id), 'cell.json'))

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

        # input
        sine_stims = [0] * n_runs
        for i_run in range(n_runs):
            sine_params, sine_stims[i_run] = get_sines(random_generator, field_pos, positions[i_run], times[i_run], dt)
        sine_stimulus = np.concatenate(sine_stims)

        tstop = (len(np.concatenate(positions))-1) * dt
        ou_process = ou_noise_input(cell, **noise_params)
        ou_process.new_seed(seed)

        # simulate
        tstop = 20000
        simulation_params = {'sec': ('soma', None), 'i_inj': np.zeros(len(sine_stimulus)), 'v_init': v_init, 'tstop': tstop,  # TODO: sine_stimulus
                             'dt': dt, 'celsius': celsius, 'onset': onset}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)

        # find all spikes
        AP_threshold = np.min(v) + 2. / 3 * np.abs(np.min(v) - np.max(v)) - 5
        onset_idxs = get_AP_onset_idxs(v, AP_threshold)

        # take window around each spike
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)
        v_APs = []
        for onset_idx in onset_idxs:
            if onset_idx - before_AP_idx >= 0 and onset_idx + after_AP_idx + 1 <= len(v):  # able to draw window
                v_AP = v[onset_idx - before_AP_idx:onset_idx + after_AP_idx + 1]
                if len(get_AP_onset_idxs(v_AP, AP_threshold)) == 1:  # no bursts
                    v_APs.append(v_AP)
        v_APs = np.vstack(v_APs)

        # STA
        spike_triggered_avg = np.mean(v_APs, 0)
        spike_triggered_std = np.std(v_APs, 0)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt

        # save and plot
        save_dir_img = os.path.join('./results/noise/STA', str(model_id))
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        print '#APs: ' + str(len(v_APs))

        pl.figure()
        pl.plot(t, v, 'k')
        pl.ylabel('Membrane potential (mV)', fontsize=16)
        pl.xlabel('Time (ms)', fontsize=16)
        pl.savefig(os.path.join(save_dir_img, 'v.svg'))

        pl.figure()
        for v_AP in v_APs:
            pl.plot(t_AP, v_AP)

        pl.figure()
        pl.plot(t_AP, spike_triggered_avg, 'r')
        pl.fill_between(t_AP, spike_triggered_avg + spike_triggered_std, spike_triggered_avg - spike_triggered_std,
                        facecolor='r', alpha=0.5)
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'STA.png'))
        pl.show()