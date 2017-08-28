from __future__ import division
import numpy as np
import os
import json
from grid_cell_stimuli.ISI_hist import get_ISI_hist, get_ISI_hists_into_outof_field, plot_ISI_hist, \
    plot_ISI_hist_into_outof_field


if __name__ == '__main__':

    folder = 'test0'
    save_dir = './results/' + folder + '/doublets'
    save_dir_data = './results/test0/data'

    # parameters
    AP_threshold = -40
    bins = np.arange(0, 200, 2)

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    position = np.load(os.path.join(save_dir_data, 'position.npy'))
    with open(os.path.join(save_dir_data, 'params.json'), 'r') as f:
        params_stim = json.load(f)
    track_len = params_stim['track_len']
    n_fields = params_stim['n_fields']

    # ISI histogram
    ISI_hist, ISIs = get_ISI_hist(v, t, AP_threshold, bins)

    # in and out field ISIs
    field_pos = np.cumsum([track_len / n_fields] * n_fields) - (track_len / n_fields) / 2
    field_pos_idxs = [np.argmin(np.abs(position-p)) for p in field_pos]
    ISI_hist_into, ISI_hist_outof = get_ISI_hists_into_outof_field(v, t, AP_threshold, bins, field_pos_idxs)

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

    plot_ISI_hist(ISI_hist, bins, save_dir)
    plot_ISI_hist_into_outof_field(ISI_hist_into, ISI_hist_outof, bins, save_dir)