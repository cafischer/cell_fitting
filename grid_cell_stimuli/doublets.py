from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
import json
from cell_characteristics.analyze_APs import get_AP_onsets


def get_ISI_hist_for_intervals(v, indices, bins):
    ISI_hist = np.zeros((len(indices), len(bins) - 1))
    for i, (s, e) in enumerate(indices):
        AP_onsets_ = get_AP_onsets(v[s:e], threshold=-20)
        ISIs_ = np.diff(t[AP_onsets_])
        ISI_hist[i, :], bin_edges = np.histogram(ISIs_, bins=bins)
    return np.sum(ISI_hist, 0)


if __name__ == '__main__':

    save_dir = './results/test0/doublets'
    save_dir_data = './results/test0/data'

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    position = np.load(os.path.join(save_dir_data, 'position.npy'))
    with open(os.path.join(save_dir_data, 'params.json'), 'r') as f:
        params_stim = json.load(f)

    # doublets
    AP_onsets = get_AP_onsets(v, threshold=-20)
    ISIs = np.diff(t[AP_onsets])
    bins = np.arange(0, 300, 2)
    ISI_hist, bin_edges = np.histogram(ISIs, bins=bins)

    n_ISI = len(ISIs)
    n_doublets = np.sum(ISI_hist[bins[:-1] < 20])
    percent_doublets = n_doublets / n_ISI
    print('Percent doublets: ', percent_doublets)

    n_theta = np.sum(ISI_hist[np.logical_and(90 <= bins[:-1], bins[:-1] < 200)])
    percent_theta = n_theta / n_ISI
    print('Percent theta: ', percent_theta)

    # in and out field ISIs
    track_len = params_stim['track_len']
    n_fields = params_stim['n_fields']
    field_pos = np.cumsum([track_len / n_fields] * n_fields) - (track_len / n_fields) / 2
    fields_pos_idx = [np.argmin(np.abs(position-p)) for p in field_pos]
    field_between = np.array([(f_j-f_i) / 2 + f_i for f_i, f_j in zip(fields_pos_idx[:-1], fields_pos_idx[1:])], dtype=int)
    into_idx = [(f_b, f_p) for f_b, f_p in zip(np.concatenate((np.array([0]), field_between)), fields_pos_idx)]
    outof_idx = [(f_p, f_b) for f_b, f_p in zip(np.concatenate((field_between, np.array([len(v)-1]))), fields_pos_idx)]

    ISI_hist_into = get_ISI_hist_for_intervals(v, into_idx, bins)
    ISI_hist_outof = get_ISI_hist_for_intervals(v, outof_idx, bins)

    # save and plots
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pl.figure()
    pl.hist(ISIs, bins=bins, color='0.5')
    pl.xlabel('ISI (ms)', fontsize=16)
    pl.ylabel('Count', fontsize=16)
    pl.savefig(os.path.join(save_dir, 'ISI_hist.svg'))
    pl.show()

    width = bins[1] - bins[0]
    pl.figure()
    pl.bar(bins[:-1], ISI_hist_into, width, color='r', alpha=0.5, label='into')
    pl.bar(bins[:-1], ISI_hist_outof, width, color='b', alpha=0.5, label='outof')
    pl.xlabel('ISI (ms)', fontsize=16)
    pl.ylabel('Count', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'ISI_hist.svg'))
    pl.show()