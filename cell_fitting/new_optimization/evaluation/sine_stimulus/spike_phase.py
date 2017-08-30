import os
import numpy as np
import json
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from grid_cell_stimuli.spike_phase import get_spike_phases, plot_phase_hist, plot_phase_vs_position_per_run, \
    compute_phase_precession, plot_phase_precession


if __name__ == '__main__':
    #save_dir = '../../../results/server/2017-07-27_09:18:59/22/L-BFGS-B/'
    save_dir = '../../../results/hand_tuning/test0'

    # load
    save_dir = os.path.join(save_dir, 'img', 'sine_stimulus', '0.5_0.2_5000_5')
    v = np.load(os.path.join(save_dir, 'v.npy'))
    t = np.load(os.path.join(save_dir, 't.npy'))
    dt = t[1] - t[0]
    i_inj = np.load(os.path.join(save_dir, 'i_inj.npy'))
    with open(os.path.join(save_dir, 'sine_params.json'), 'r') as f:
        sine_params = json.load(f)

    # parameter
    AP_threshold = -30
    order = int(round(20 / dt))
    dist_to_AP = int(round(250 / dt))
    speed = 0.040  # cm/ms

    # rebuild theta stim (sine2)
    x = np.arange(0, sine_params['sine1_dur'] + sine_params['dt'], sine_params['dt'])
    theta = sine_params['amp2'] * np.sin(2 * np.pi * x * sine_params['freq2']/1000)
    onset = np.zeros(int(round(sine_params['onset_dur']/sine_params['dt'])))
    offset = np.zeros(int(round(sine_params['offset_dur']/sine_params['dt'])))
    theta = np.concatenate((onset, theta, offset))

    # spike phase
    AP_onsets = get_AP_onset_idxs(v, threshold=AP_threshold)
    phases = get_spike_phases(AP_onsets, t, theta, order, dist_to_AP)
    not_nan = np.logical_not(np.isnan(phases))
    phases = phases[not_nan]
    AP_onsets = AP_onsets[not_nan]
    plot_phase_hist(phases, save_dir)

    # phase precession
    position = t * speed
    track_len = position[-1]
    phases_pos = position[AP_onsets]
    run_start_idx = [0, len(position)]
    plot_phase_vs_position_per_run(phases, phases_pos, AP_onsets, track_len, run_start_idx, save_dir)

    slope, intercept, best_shift = compute_phase_precession(phases, phases_pos)
    plot_phase_precession(phases, phases_pos, slope, intercept, best_shift, save_dir, show=True)