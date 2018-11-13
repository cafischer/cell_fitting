from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as pl
import json
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.evaluation.plot_sine_stimulus import simulate_sine_stimulus
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx
from scipy.stats import circmean, circstd
from grid_cell_stimuli.spike_phase import get_spike_phases_by_min, plot_phase_hist, plot_phase_vs_position_per_run, \
    compute_phase_precession, plot_phase_precession


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'
    model_ids = [2] #range(1, 7)
    mechanism_dir = '../../../model/channels/vavoulis'
    load_mechanism_dir(mechanism_dir)

    amp1 = 0.4  # 0.5
    amp2 = 0.4  # 0.2
    freq1 = 0.1  # 0.5: 1000, 0.25: 2000, 0.1: 5000, 0.05: 10000
    sine1_dur = 1./freq1 * 1000 / 2.
    freq2 = 5  # 5  # 20
    onset_dur = offset_dur = 500
    dt = 0.01

    for model_id in model_ids:
        # load model
        model_dir = os.path.join(save_dir, str(model_id), 'cell_rounded.json')
        cell = Cell.from_modeldir(model_dir)

        save_dir_img = os.path.join(save_dir, str(model_id), 'img', 'sine_stimulus', 'traces',
                                    str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2), 'phase_hist')
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        # simulate
        v, t, i_inj = simulate_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)

        # parameter
        AP_threshold = -10
        order = to_idx(20, dt)
        dist_to_AP = to_idx(1./freq2 * 1000, dt)

        # rebuild theta stim (sine2)
        x = np.arange(0, sine1_dur + dt, dt)
        theta = amp2 * np.sin(2 * np.pi * x * freq2 / 1000)
        onset = np.zeros(to_idx(onset_dur, dt))
        offset = np.zeros(to_idx(offset_dur, dt))
        theta = np.concatenate((onset, theta, offset))

        # spike phase
        AP_onset_idxs = get_AP_onset_idxs(v, threshold=AP_threshold)
        AP_onset_idxs = np.concatenate((AP_onset_idxs, np.array([-1])))
        AP_max_idxs = np.array([get_AP_max_idx(v, i, j) for i, j in zip(AP_onset_idxs[:-1], AP_onset_idxs[1:])])
        phases = get_spike_phases_by_min(AP_max_idxs, t, theta, order, dist_to_AP)
        t_phases = t[AP_max_idxs]
        not_nan = np.logical_not(np.isnan(phases))
        phases = phases[not_nan]
        t_phases = t_phases[not_nan]
        AP_max_idxs = AP_max_idxs[not_nan]
        mean_phase = circmean(phases, 360, 0)
        std_phase = circstd(phases, 360, 0)
        slope, intercept, best_shift = compute_phase_precession(phases, t_phases)

        sine_dict = dict(phases=list(phases), t_phases=list(t_phases), mean_phase=[mean_phase], std_phase=[std_phase],
                         slope=slope, intercept=intercept)
        with open(os.path.join(save_dir_img, 'sine_dict.json'), 'w') as f:
            json.dump(sine_dict, f)

        # parameter
        AP_threshold = -10
        order = to_idx(20, dt)
        speed = 0.040  # cm/ms

        # phase precession
        print slope
        plot_phase_hist(phases, mean_phase=mean_phase, std_phase=std_phase, save_dir_img=save_dir_img)
        plot_phase_vs_position_per_run(phases, t_phases, AP_max_idxs, t[-1], [0, len(t)], save_dir_img)
        plot_phase_precession(phases, t_phases, slope, intercept, best_shift, save_dir_img)

        pl.figure()
        pl.plot(t, v)

        pl.show()