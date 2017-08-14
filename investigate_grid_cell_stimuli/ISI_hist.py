from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
import json
from cell_characteristics.analyze_APs import get_AP_onset_idxs


def get_ISI_hist(v, t, AP_threshold, bins):
    AP_onsets = get_AP_onset_idxs(v, threshold=AP_threshold)
    ISIs = np.diff(t[AP_onsets])
    ISI_hist, bin_edges = np.histogram(ISIs, bins=bins)
    return ISI_hist, ISIs


def get_ISI_hists_into_outof_field(AP_threshold, field_pos_idxs):
    field_between = np.array([(f_j - f_i) / 2 + f_i for f_i, f_j in zip(field_pos_idxs[:-1], field_pos_idxs[1:])],
                             dtype=int)
    into_idx = [(f_b, f_p) for f_b, f_p in zip(np.concatenate((np.array([0]), field_between)), field_pos_idxs)]
    outof_idx = [(f_p, f_b) for f_b, f_p in
                 zip(np.concatenate((field_between, np.array([len(v) - 1]))), field_pos_idxs)]
    ISI_hist_into = get_ISI_hist_for_intervals(v, AP_threshold, into_idx, bins)
    ISI_hist_outof = get_ISI_hist_for_intervals(v, AP_threshold, outof_idx, bins)
    return ISI_hist_into, ISI_hist_outof


def get_ISI_hist_for_intervals(v, AP_threshold, indices, bins):
    ISI_hist = np.zeros((len(indices), len(bins) - 1))
    for i, (s, e) in enumerate(indices):
        AP_onsets_ = get_AP_onset_idxs(v[s:e], threshold=AP_threshold)
        ISIs_ = np.diff(t[AP_onsets_])
        ISI_hist[i, :], bin_edges = np.histogram(ISIs_, bins=bins)
    return np.sum(ISI_hist, 0)


if __name__ == '__main__':

    save_dir = './results/test0/doublets'
    save_dir_data = './results/test0/data'
    AP_threshold = -40

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    position = np.load(os.path.join(save_dir_data, 'position.npy'))
    with open(os.path.join(save_dir_data, 'params.json'), 'r') as f:
        params_stim = json.load(f)
    track_len = params_stim['track_len']
    n_fields = params_stim['n_fields']

    # ISI histogram
    bins = np.arange(0, 200, 2)
    ISI_hist, ISIs = get_ISI_hist(v, t, AP_threshold, bins)

    # in and out field ISIs
    field_pos = np.cumsum([track_len / n_fields] * n_fields) - (track_len / n_fields) / 2
    field_pos_idxs = [np.argmin(np.abs(position-p)) for p in field_pos]
    ISI_hist_into, ISI_hist_outof = get_ISI_hists_into_outof_field(AP_threshold, field_pos_idxs)

    # save and plots
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_ISI = len(ISIs)
    n_doublets = np.sum(ISI_hist[bins[:-1] < 20])
    percent_doublets = n_doublets / n_ISI
    print('Percent doublets: ', percent_doublets)

    n_theta = np.sum(ISI_hist[np.logical_and(90 <= bins[:-1], bins[:-1] < 200)])
    percent_theta = n_theta / n_ISI
    print('Percent theta: ', percent_theta)

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