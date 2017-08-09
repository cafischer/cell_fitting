from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import pandas as pd
import json
from scipy.signal import argrelmax
from cell_characteristics.analyze_APs import get_AP_onsets


if __name__ == '__main__':
    save_dir = './results/test0/spike_theta_phase'
    save_dir_data = './results/test0/data'
    save_dir_theta = './results/test0/ramp_and_theta'

    # params
    freq = 8
    freq2 = 5
    freq3 = 11

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    position = np.load(os.path.join(save_dir_data, 'position.npy'))
    with open(os.path.join(save_dir_data, 'params.json'), 'r') as f:
        params_data = json.load(f)
    track_len = params_data['track_len']
    dt = t[1] - t[0]
    dt_sec = dt / 1000
    theta = np.load(os.path.join(save_dir_theta, 'theta.npy'))

    # extract phase
    AP_onsets = get_AP_onsets(v, threshold=-40)
    phases_pos = position[AP_onsets]
    phases = np.zeros(len(AP_onsets))

    for i, AP_idx in enumerate(AP_onsets):
        order = int(round(20 / dt))
        dist_to_AP = int(round(200 / dt))
        peak_before_idx = argrelmax(theta[AP_idx-dist_to_AP:AP_idx], order=order)[0][-1] + AP_idx - dist_to_AP
        peak_after_idx = argrelmax(theta[AP_idx:AP_idx+dist_to_AP], order=order)[0][0] + AP_idx
        phases[i] = 360 * (t[AP_idx] - t[peak_before_idx]) / (t[peak_after_idx] - t[peak_before_idx])

        # print phases[i]
        # pl.figure()
        # pl.plot(t[int(peak_before_idx-10):int(peak_after_idx+10)],
        #         theta[int(peak_before_idx-10):int(peak_after_idx+10)], 'b')
        # pl.plot(t[peak_before_idx], theta[peak_before_idx], 'og')
        # pl.plot(t[peak_after_idx], theta[peak_after_idx], 'or')
        # pl.plot(t[AP_idx], theta[AP_idx], 'oy')
        # pl.show()

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'phases.npy'), phases)
    np.save(os.path.join(save_dir, 'phases_pos.npy'), phases_pos)

    pl.figure()
    pl.hist(phases, bins=np.arange(0, 360 + 10, 10), weights=np.ones(len(phases))/len(phases), color='0.5')
    pl.xlabel('Phase ($^{\circ}$)', fontsize=16)
    pl.ylabel('Normalized Count', fontsize=16)
    pl.xlim(0, 360)
    pl.savefig(os.path.join(save_dir, 'phase_hist.svg'))
    pl.show()

    run_start_idx = np.where(np.diff(position) < 0)[0]
    run_start_idx = np.concatenate((np.array([0]), run_start_idx))
    phases_run = [0] * (len(run_start_idx) - 1)
    phases_pos_run = [0] * (len(run_start_idx) - 1)
    for i_run, (run_start, run_end) in enumerate(zip(run_start_idx[:-1], run_start_idx[1:])):
        phases_run[i_run] = phases[np.logical_and(AP_onsets > run_start, AP_onsets < run_end)]
        phases_pos_run[i_run] = phases_pos[np.logical_and(AP_onsets > run_start, AP_onsets < run_end)]

    pl.figure()
    for i_run in range(len(phases_run)):
        pl.plot(phases_pos_run[i_run], phases_run[i_run], 'o')
    pl.xlim(0, track_len)
    pl.ylim(0, 360)
    pl.xlabel('Position (cm)', fontsize=16)
    pl.ylabel('Phase ($^{\circ}$)', fontsize=16)
    pl.savefig(os.path.join(save_dir, 'phase_vs_position.svg'))
    pl.show()