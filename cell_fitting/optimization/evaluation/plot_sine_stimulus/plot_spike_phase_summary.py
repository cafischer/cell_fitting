from __future__ import division
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pl
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.evaluation.plot_sine_stimulus import simulate_sine_stimulus
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from scipy.stats import circmean, circstd
from grid_cell_stimuli.spike_phase import get_spike_phases
pl.style.use('paper')


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'
    model_ids = range(1, 7)
    mechanism_dir = '../../../model/channels/vavoulis'
    load_mechanism_dir(mechanism_dir)

    amp1 = 0.6  # 0.5
    amp2 = 0.2  # 0.2
    amp1_data = None
    amp2_data = None
    freq1 = 0.1  # 0.5: 1000, 0.25: 2000, 0.1: 5000, 0.05: 10000
    sine1_dur = 1./freq1 * 1000 / 2
    freq2 = 5  # 5  # 20
    onset_dur = offset_dur = 500
    dt = 0.01
    save_dir_data = os.path.join('../../../data/plots/sine_stimulus/traces/', 'rat', 'summary', 'spike_phase',
                                 str(amp1_data)+'_'+str(amp2_data)+'_'+str(freq1)+'_'+str(freq2))
    save_dir_img = os.path.join(save_dir, 'img', 'sine_stimulus', 'phase_hist',
                                str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2))

    phase_means_models = []
    phase_stds_model = []
    for model_id in model_ids:
        # load model
        model_dir = os.path.join(save_dir, str(model_id), 'cell.json')
        cell = Cell.from_modeldir(model_dir)

        # simulate
        v, t, i_inj = simulate_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)

        # parameter
        AP_threshold = 0
        order = to_idx(20, dt)
        dist_to_AP = to_idx(250, dt)

        # rebuild theta stim (sine2)
        x = np.arange(0, sine1_dur + dt, dt)
        theta = amp2 * np.sin(2 * np.pi * x * freq2/1000)
        onset = np.zeros(to_idx(onset_dur, dt))
        offset = np.zeros(to_idx(offset_dur, dt))
        theta = np.concatenate((onset, theta, offset))

        # spike phase
        AP_onsets = get_AP_onset_idxs(v, threshold=AP_threshold)
        phases = get_spike_phases(AP_onsets, t, theta, order, dist_to_AP)
        not_nan = np.logical_not(np.isnan(phases))
        phases = phases[not_nan]
        AP_onsets = AP_onsets[not_nan]
        mean_phase = circmean(phases, 360, 0)
        std_phase = circstd(phases, 360, 0)
        phase_means_models.append(mean_phase)
        phase_stds_model.append(std_phase)

    phase_means_data = np.load(os.path.join(save_dir_data, 'phase_means.npy'))
    phase_stds_data = np.load(os.path.join(save_dir_data, 'phase_stds.npy'))

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)
    data = pd.DataFrame(np.array([phase_means_data, phase_stds_data]).T, columns=['Mean Phase', 'Std Phase'])
    jp = sns.jointplot('Mean Phase', 'Std Phase', data=data, stat_func=None, color='k', alpha=0.5)
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = phase_means_models
    jp.y = phase_stds_model
    jp.plot_joint(pl.scatter, c='r', alpha=0.5)
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(phase_means_models[i] + 0.5, phase_stds_model[i] + 0.5), color='r', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'mean_std_phase_hist.png'))
    pl.show()