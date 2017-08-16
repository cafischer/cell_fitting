from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import json
import os
import pandas as pd


if __name__ == '__main__':
    folder = 'test0'
    save_dir = './results/' + folder + '/in_out_field_diff'
    save_dir_data = './results/' + folder + '/data'
    save_dir_ramp_and_theta = './results/test0/ramp_and_theta'
    save_dir_in_out_fields = './results/test0/in_out_fields'

    # load
    ramp = np.load(os.path.join(save_dir_ramp_and_theta, 'ramp.npy'))
    theta_envelope = np.load(os.path.join(save_dir_ramp_and_theta, 'theta_envelope.npy'))
    t = np.load(os.path.join(save_dir_ramp_and_theta, 't.npy'))
    dt = t[1] - t[0]
    in_field = np.load(os.path.join(save_dir_in_out_fields, 'in_field.npy')).astype(bool)
    out_field = np.load(os.path.join(save_dir_in_out_fields, 'out_field.npy')).astype(bool)
    position = np.load(os.path.join(save_dir_data, 'position.npy'))
    with open(os.path.join(save_dir_in_out_fields, 'params.json'), 'r') as f:
        params_in_out_fields = json.load(f)
    bin_size = params_in_out_fields['bin_size']

    # bin ramp and theta
    bins = np.arange(bin_size, np.max(position) + bin_size, bin_size)
    n_bins = len(bins) - 1  # -1 for last bin edge
    run_start_idx = np.where(np.diff(position) < 0)[0]
    pos_runs = np.split(position, run_start_idx)
    ramp_runs = np.split(ramp, run_start_idx)
    theta_runs = np.split(theta_envelope, run_start_idx)

    ramp_in_field_per_run = []
    ramp_out_field_per_run = []
    theta_in_field_per_run = []
    theta_out_field_per_run = []
    for i_run, (pos_run, ramp_run, theta_run) in enumerate(zip(pos_runs, ramp_runs, theta_runs)):
        pos_binned = np.digitize(pos_run, bins)
        pos_in_track = np.unique(pos_binned)[np.unique(pos_binned) < n_bins - 1]  # to ignore data higher than track_len
        ramp_per_run = pd.Series(ramp_run).groupby(pos_binned).mean()[pos_in_track]
        theta_per_run = pd.Series(theta_run).groupby(pos_binned).mean()[pos_in_track]
        ramp_in_field_per_run.extend(ramp_per_run[in_field[pos_in_track]].values)
        ramp_out_field_per_run.extend(ramp_per_run[out_field[pos_in_track]].values)
        theta_in_field_per_run.extend(theta_per_run[in_field[pos_in_track]].values)
        theta_out_field_per_run.extend(theta_per_run[out_field[pos_in_track]].values)

    # in and out field difference
    ramp_in_mean = np.mean(ramp_in_field_per_run)
    ramp_out_mean = np.mean(ramp_out_field_per_run)
    theta_in_mean = np.mean(theta_in_field_per_run)
    theta_out_mean = np.mean(theta_out_field_per_run)
    ramp_diff = ramp_in_mean - ramp_out_mean
    theta_diff = theta_in_mean - theta_out_mean

    # save and plots
    print('$\Delta$Ramp: ', ramp_diff)  # exp: 2.9 +- 0.3 mV
    print('$\Delta$Theta: ', theta_diff)  # exp:  0.72 +- 0.12 mV