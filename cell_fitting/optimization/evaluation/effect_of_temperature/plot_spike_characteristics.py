from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as pl
from cell_fitting.optimization.evaluation.rampIV import simulate_rampIV
from nrn_wrapper import Cell, load_mechanism_dir
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_spike_characteristics
from cell_fitting.optimization.evaluation.effect_of_temperature import set_q10, set_q10_g
pl.style.use('paper')


# save dir
save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'
mechanism_dir = '../../../model/channels/vavoulis_with_temp'
data_dir = '../../../data/plots/spike_characteristics/distributions/rat'
celsius = 22
q10 = 1.0
q10_g = 1.0

# for models
protocol = 'rampIV'
model_ids = np.arange(1, 7)
sweep_idxs = range(83)  # till 4.0 nA
v_rest_shift = -16
tstop = 161.99
dt = 0.01
AP_threshold = -20  # mV
AP_interval = 8.0  # TODO 2.5  # ms
fAHP_interval = 15.0  # TODO: 4.0
AP_width_before_onset = 2  # ms
DAP_interval = 50  # TODO: 10  # ms
order_fAHP_min = 1.0  # ms (how many points to consider for the minimum)
order_DAP_max = 1.0  # ms (how many points to consider for the minimum)
min_dist_to_DAP_max = 0.5  # ms
k_splines = 3
s_splines = None
return_characteristics = ['AP_amp', 'AP_width', 'fAHP_amp', 'DAP_amp', 'DAP_deflection', 'DAP_width', 'DAP_time',
                          'fAHP2DAP_time']
load_mechanism_dir(mechanism_dir)

spike_characteristics_list = []
v_list = []


for model_id in model_ids:
    save_dir_model = save_dir + str(model_id)
    model_dir = os.path.join(save_dir_model, 'cell.json')
    cell = Cell.from_modeldir(model_dir)
    set_q10(cell, q10)
    set_q10_g(cell, q10_g)

    for ramp_amp in np.arange(3.0, 4.0, 0.1):
        v, t, i_inj = simulate_rampIV(cell, ramp_amp, v_init=-75, celsius=celsius)

        onset_idxs = get_AP_onset_idxs(v, AP_threshold)
        if len(onset_idxs) == 1 and onset_idxs * dt < 12.5:
            # get spike characteristics
            print model_id, ramp_amp
            start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
            std_idx_times = (0, start_i_inj * dt)
            v_rest = np.mean(v[0:start_i_inj])
            spike_characteristics = np.array(get_spike_characteristics(v, t, return_characteristics, v_rest,
                                                                       AP_threshold, AP_interval, AP_width_before_onset,
                                                                       fAHP_interval, std_idx_times,
                                                                       k_splines, s_splines, order_fAHP_min,
                                                                       DAP_interval, order_DAP_max, min_dist_to_DAP_max,
                                                                       check=True), dtype=float)
            spike_characteristics_list.append(spike_characteristics)
            v_list.append(v)
            break
characteristics_mat_models = np.vstack(spike_characteristics_list)  # candidates vs characteristics
AP_matrix_models = np.vstack(v_list)  # candidates vs t

# for exp. cells
characteristics_mat = np.load(os.path.join(data_dir, 'characteristics_mat.npy'))
candidate_mat = np.load(os.path.join(data_dir, 'AP_mat.npy'))

for i, characteristic in enumerate(return_characteristics):
    min_val = min(np.nanmin(characteristics_mat[:, i]), np.nanmin(characteristics_mat_models[:, i]))
    max_val = max(np.nanmax(characteristics_mat[:, i]), np.nanmax(characteristics_mat_models[:, i]))
    bins = np.linspace(min_val, max_val, 100)
    if return_characteristics[i] == 'AP_width':
        bins = np.arange(min_val, max_val+0.015, 0.015)

    hist, bins = np.histogram(characteristics_mat[~np.isnan(characteristics_mat[:, i]), i], bins=bins)
    hist_models, _ = np.histogram(characteristics_mat_models[~np.isnan(characteristics_mat_models[:, i]), i],
                                  bins=bins)
    character_name_dict = {'AP_amp': 'AP Amplitude (mV)', 'AP_width': 'AP Width (ms)',
                           'fAHP_amp': 'fAHP Amplitude (mV)',
                           'DAP_amp': 'DAP Amplitude (mV)', 'DAP_deflection': 'DAP Deflection (mV)',
                           'DAP_width': 'DAP Width (ms)', 'DAP_time': 'DAP Time (ms)',
                           'fAHP2DAP_time': '$Time_{DAP_{max} - fAHP_{min}} (ms)$'}

    # plot
    save_dir_img = os.path.join(save_dir, 'img', protocol, 'spike_characteristics')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    print 'Models nan: ' + str(model_ids[np.isnan(characteristics_mat_models[:, i])])
    np.savetxt(os.path.join(save_dir_img, 'characteristics_mat.txt'), characteristics_mat_models, fmt='%.3f')

    fig, ax1 = pl.subplots()
    ax1.bar(bins[:-1], hist / np.shape(characteristics_mat)[0], width=bins[1] - bins[0], color='k', alpha=0.5)
    ax1.set_ylim(0, 0.1)
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True)
    bars = ax2.bar(bins[:-1], hist_models/len(model_ids), width=bins[1] - bins[0], color='r', alpha=0.5)

    # put model numbers above bars
    model_bin = np.digitize(characteristics_mat_models[:, i], bins) - 1
    if np.any(np.isnan(characteristics_mat_models[:, i])):
        model_bin = np.array(model_bin, dtype=float)
        model_bin[np.isnan(characteristics_mat_models[:, i])] = np.nan
    for m_bin in np.unique(model_bin[~np.isnan(model_bin)]):
        h = np.sum(m_bin == model_bin) / len(model_ids)
        s = ''
        model_idxs = np.where(m_bin == model_bin)[0]
        for j, idx in enumerate(model_idxs):
            if j == len(model_idxs)-1:
                s += str(model_ids[idx])
            else:
                s += str(model_ids[idx]) + ','
        dbin = np.diff(bins)[0]
        ax2.annotate(s, xy=(bins[int(m_bin)]-dbin, h + 0.01), color='r', fontsize=8)

    ax2.set_ylim(0, 1)
    ax1.set_xlabel(character_name_dict.get(return_characteristics[i], return_characteristics[i]))
    ax1.set_ylabel('Proportion Exp. Cells')
    ax2.set_ylabel('Proportion Models', fontsize=18, color='r')
    pl.tight_layout()
    #pl.savefig(os.path.join(save_dir_img, 'hist_' + return_characteristics[i] + '.png'))
    pl.show()