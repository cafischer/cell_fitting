from __future__ import division
import os
import numpy as np
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_characteristics import to_idx
from grid_cell_stimuli.spike_phase import get_spike_phases, plot_phase_hist, plot_phase_vs_position_per_run, \
    compute_phase_precession, plot_phase_precession
from cell_fitting.data.data_sinus_mat import find_sine_trace
from scipy.stats import circmean, circstd
from cell_fitting.optimization.evaluation import plot_v
import matplotlib.pyplot as pl


if __name__ == '__main__':
    animal = 'rat'
    save_dir = os.path.join('../plots/sine_stimulus/traces/', animal)
    amp1_use = None #0.6
    amp2_use = None #0.2
    freq1 = 0.1
    freq2 = 5
    onset_dur = offset_dur = 500
    v_mat, t_mat, cell_ids, amp1s, amp2s, freq1s, freq2s = find_sine_trace(amp1_use, amp2_use, freq1, freq2)

    phase_means = []
    phase_stds = []

    for v, t, cell_id, amp1, amp2 in zip(v_mat, t_mat, cell_ids, amp1s, amp2s):
        dt = t[1] - t[0]

        # parameter
        AP_threshold = 0
        order = to_idx(20, dt)
        dist_to_AP = to_idx(250, dt)
        speed = 0.040  # cm/ms

        # rebuild theta stim (sine2)
        dur1 = 1./freq1 * 1000 / 2
        x = np.arange(0, dur1 + dt, dt)
        theta = amp2 * np.sin(2 * np.pi * x * freq2/1000)
        onset = np.zeros(to_idx(onset_dur, dt))
        offset = np.zeros(to_idx(offset_dur, dt))
        theta = np.concatenate((onset, theta, offset))

        # save and plots
        save_dir_img = os.path.join(save_dir, cell_id, str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2),
                                    'spike_phase')
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        plot_v(t, v, c='k',
               save_dir_img=os.path.join(save_dir, cell_id, str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2)))

        # spike phase
        AP_onsets = get_AP_onset_idxs(v, threshold=AP_threshold)
        phases = get_spike_phases(AP_onsets, t, theta, order, dist_to_AP)
        not_nan = np.logical_not(np.isnan(phases))
        phases = phases[not_nan]
        AP_onsets = AP_onsets[not_nan]
        mean_phase = circmean(phases, 360, 0)
        std_phase = circstd(phases, 360, 0)
        phase_means.append(mean_phase)
        phase_stds.append(std_phase)
        plot_phase_hist(phases, save_dir_img, mean_phase=mean_phase, std_phase=std_phase, show=False)

        # phase precession
        position = t * speed
        track_len = position[-1]
        phases_pos = position[AP_onsets]
        run_start_idx = [0, len(position)]
        plot_phase_vs_position_per_run(phases, phases_pos, AP_onsets, track_len, run_start_idx, save_dir_img)

        slope, intercept, best_shift = compute_phase_precession(phases, phases_pos)
        plot_phase_precession(phases, phases_pos, slope, intercept, best_shift, save_dir_img)
        pl.show()
        pl.close()

    # save summary
    save_dir_summary = os.path.join(save_dir, 'summary', 'spike_phase',
                                    str(amp1_use)+'_'+str(amp2_use)+'_'+str(freq1)+'_'+str(freq2))
    if not os.path.exists(save_dir_summary):
        os.makedirs(save_dir_summary)
    np.save(os.path.join(save_dir_summary, 'phase_means.npy'), phase_means)
    np.save(os.path.join(save_dir_summary, 'phase_stds.npy'), phase_stds)
    np.save(os.path.join(save_dir_summary, 'amp1s.npy'), amp1s)
    np.save(os.path.join(save_dir_summary, 'amp2s.npy'), amp2s)
