from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
import json
from cell_characteristics.analyze_APs import get_AP_onsets


if __name__ == '__main__':

    save_dir = './results/test0/doublets'
    save_dir_data = './results/test0/data'

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))

    # doublets
    AP_onsets = get_AP_onsets(v, threshold=-20)
    ISIs = np.diff(t[AP_onsets])
    bins = np.arange(0, 300, 2)
    ISI_hist, bin_edges = np.histogram(ISIs, bins=bins)

    n_ISI = len(ISIs)
    n_doublets = np.sum(ISI_hist[bins[:-1] < 20])
    percent_doublets = n_doublets / n_ISI
    print('percent doublets: ', percent_doublets)

    n_theta = np.sum(ISI_hist[np.logical_and(90 <= bins[:-1], bins[:-1] < 200)])
    percent_theta = n_theta / n_ISI
    print('percent theta: ', percent_theta)

    # in and out field ISIs
    params_stim = np.load()
    position = np.load()
    t = np.load()
    track_len = params_stim['track_len']
    n_fields = params_stim['n_fields']
    field_pos = np.cumsum([track_len / n_fields] * n_fields) - (track_len / n_fields) / 2
    fields_pos_idx = [np.argmin(np.abs(position-p)) for p in field_pos]
    field_between = np.array([(f_j-f_i) / 2 + f_i for f_i, f_j in zip(fields_pos_idx[:-1], fields_pos_idx[1:])])
    into_idx = [(f_b, f_p) for f_b, f_p in zip(np.concatenate((np.array([0]), field_between)), fields_pos_idx)]
    outof_idx = [(f_p, f_b) for f_b, f_p in zip(np.concatenate((field_between, np.array([-1]))), fields_pos_idx)]
    # TODO

    # save and plots
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pl.figure()
    pl.hist(ISIs, bins=bins, color='0.5')
    pl.xlabel('ISI (ms)', fontsize=16)
    pl.ylabel('Count', fontsize=16)
    pl.savefig(os.path.join(save_dir, 'ISI_hist.png'))
    pl.show()