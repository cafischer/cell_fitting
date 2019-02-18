from __future__ import division
import os
import numpy as np
import json
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx
from cell_characteristics import to_idx
from grid_cell_stimuli.spike_phase import get_spike_phases_by_min, plot_phase_hist, plot_phase_vs_position_per_run, \
    compute_phase_precession, plot_phase_precession
from cell_fitting.data.data_sinus_mat import find_sine_trace
from scipy.stats import circmean, circstd
from cell_fitting.optimization.evaluation import plot_v
import matplotlib.pyplot as pl
from cell_fitting.read_heka import shift_v_rest
from cell_fitting.data import check_cell_has_DAP
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_fitting.optimization.evaluation.plot_sine_stimulus.plot_spike_phase import get_theta_and_phases


if __name__ == '__main__':
    protocol = 'sine_stimulus'
    animal = 'rat'
    amp1_use = None
    amp2_use = None
    freq1 = 0.1
    freq2 = 5
    sine1_dur = 1. / freq1 * 1000 / 2.
    onset_dur = offset_dur = 500
    v_shift = -16
    save_dir = os.path.join('../plots', protocol, 'traces', animal)

    v_mat, t_mat, cell_ids, amp1s, amp2s, freq1s, freq2s = find_sine_trace(amp1_use, amp2_use, freq1, freq2)
    cell_ids_new = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)
    cell_ids_new = filter(lambda id: check_cell_has_DAP(id), cell_ids_new)
    cell_ids_new = np.unique(cell_ids_new)  # no doubles
    cell_id_idxs = np.array([np.where(c_id == np.array(cell_ids))[0][0] for c_id in cell_ids_new])
    v_mat = v_mat[cell_id_idxs]
    t_mat = t_mat[cell_id_idxs]
    amp1s = np.array(amp1s)[cell_id_idxs]
    amp2s = np.array(amp2s)[cell_id_idxs]
    freq1s = np.array(freq1s)[cell_id_idxs]
    freq2s = np.array(freq2s)[cell_id_idxs]
    cell_ids = np.array(cell_ids)[cell_id_idxs]

    phase_means = []
    phase_stds = []
    for v, t, cell_id, amp1, amp2 in zip(v_mat, t_mat, cell_ids, amp1s, amp2s):
        v = shift_v_rest(v, v_shift)
        dt = t[1] - t[0]

        # parameter
        AP_threshold = 0
        order = to_idx(20, dt)
        dist_to_AP = to_idx(250, dt)

        # rebuild theta stim (sine2)
        theta, phases_theta = get_theta_and_phases(sine1_dur, amp2, freq2, onset_dur, offset_dur, dt)

        # save and plots
        save_dir_img = os.path.join(save_dir, cell_id, str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2),
                                    'spike_phase')
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        # plot_v(t, v, c='k',
        #        save_dir_img=os.path.join(save_dir, cell_id, str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2)))

        # spike phase
        AP_onset_idxs = get_AP_onset_idxs(v, threshold=AP_threshold)
        AP_onset_idxs = np.concatenate((AP_onset_idxs, np.array([-1])))
        AP_max_idxs = np.array([get_AP_max_idx(v, i, j) for i, j in zip(AP_onset_idxs[:-1], AP_onset_idxs[1:])])
        phases = phases_theta[AP_max_idxs]
        #phases = get_spike_phases_by_min(AP_max_idxs, t, theta, order, dist_to_AP)
        t_phases = t[AP_max_idxs]
        not_nan = np.logical_not(np.isnan(phases))
        phases = phases[not_nan]
        t_phases = t_phases[not_nan]
        AP_max_idxs = AP_max_idxs[not_nan]
        mean_phase = circmean(phases, 360, 0)
        std_phase = circstd(phases, 360, 0)
        phase_means.append(mean_phase)
        phase_stds.append(std_phase)
        slope, intercept, best_shift = compute_phase_precession(phases, t_phases)

        # save
        sine_dict = dict(phases=list(phases), t_phases=list(t_phases), mean_phase=[mean_phase], std_phase=[std_phase],
                         slope=slope, intercept=intercept)
        with open(os.path.join(save_dir_img, 'sine_dict.json'), 'w') as f:
            json.dump(sine_dict, f)
        np.save(os.path.join(save_dir, cell_id, str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2), 't.npy'), t)
        np.save(os.path.join(save_dir, cell_id, str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2), 'v.npy'), v)


        plot_phase_hist(phases, mean_phase=mean_phase, std_phase=std_phase, save_dir_img=save_dir_img)
        # plot_phase_vs_position_per_run(phases, t_phases, AP_onsets, t[-1], [0, len(t)], save_dir_img)
        # plot_phase_precession(phases, t_phases, slope, intercept, best_shift, save_dir_img)
        # pl.show()
        pl.close('all')

    # save summary
    save_dir_summary = os.path.join(save_dir, 'summary', 'spike_phase',
                                    str(amp1_use)+'_'+str(amp2_use)+'_'+str(freq1)+'_'+str(freq2))
    if not os.path.exists(save_dir_summary):
        os.makedirs(save_dir_summary)
    np.save(os.path.join(save_dir_summary, 'phase_means.npy'), phase_means)
    np.save(os.path.join(save_dir_summary, 'phase_stds.npy'), phase_stds)
    np.save(os.path.join(save_dir_summary, 'amp1s.npy'), amp1s)
    np.save(os.path.join(save_dir_summary, 'amp2s.npy'), amp2s)
