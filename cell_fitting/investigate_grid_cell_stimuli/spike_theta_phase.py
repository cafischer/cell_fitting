from __future__ import division
import numpy as np
import os
import json
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from grid_cell_stimuli.spike_phase import get_spike_phases, plot_phase_hist, plot_phase_vs_position_per_run


if __name__ == '__main__':
    folder = 'test0'
    save_dir = './results/' + folder + '/spike_theta_phase'
    save_dir_data = './results/' + folder + '/data'
    save_dir_theta = './results/' + folder + '/ramp_and_theta'

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    dt = t[1] - t[0]
    position = np.load(os.path.join(save_dir_data, 'position.npy'))
    with open(os.path.join(save_dir_data, 'params.json'), 'r') as f:
        params_data = json.load(f)
    track_len = params_data['track_len']
    theta = np.load(os.path.join(save_dir_theta, 'theta.npy'))

    # params
    AP_threshold = -40
    order = int(round(20 / dt))
    dist_to_AP = int(round(200 / dt))

    # extract phase
    AP_onsets = get_AP_onset_idxs(v, threshold=AP_threshold)
    phases_pos = position[AP_onsets]
    phases = get_spike_phases(AP_onsets, t, theta, order, dist_to_AP)

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'phases.npy'), phases)
    np.save(os.path.join(save_dir, 'phases_pos.npy'), phases_pos)

    plot_phase_hist(phases, save_dir)
    run_start_idx = np.where(np.diff(position) < 0)[0]
    run_start_idx = np.concatenate((np.array([0]), run_start_idx))
    plot_phase_vs_position_per_run(phases, phases_pos, AP_onsets, track_len, run_start_idx, save_dir)