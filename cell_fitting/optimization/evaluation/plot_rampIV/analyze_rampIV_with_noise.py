from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.optimization.evaluation.plot_rampIV import simulate_rampIV, plot_rampIV
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.investigate_grid_cell_stimuli.model_noise.with_OU import ou_noise_input
from cell_characteristics.analyze_APs import get_AP_onset_idxs
pl.style.use('paper')


def find_current_threshold_with_noise(cell, ramp_amps, n_trials=20):
    AP_probs = np.zeros(len(ramp_amps))
    for ramp_amp_idx, ramp_amp in enumerate(ramp_amps):
        for trial in range(n_trials):
            v, t, i_inj = simulate_rampIV(cell, ramp_amp)
            start = np.where(i_inj)[0][0]
            onset_idxs = get_AP_onset_idxs(v[start:], threshold=0)
            if len(onset_idxs) >= 1:
                AP_probs[ramp_amp_idx] += 1
    AP_probs /= n_trials
    return AP_probs


if __name__ == '__main__':
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    #noise_params = {'g_e0': 0.001, 'g_i0': 0.05, 'std_e': 0.008, 'std_i': 0.008, 'tau_e': 2.5, 'tau_i': 5.0}
    noise_params = {'g_e0': 0, 'g_i0': 0.05, 'std_e': 0, 'std_i': 0, 'tau_e': 2.5, 'tau_i': 5.0}

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    ou_process = ou_noise_input(cell, **noise_params)

    # current to elicit AP
    # min_AP_prob = 0.75
    # ramp_amps = np.arange(0.5, 4.0 + 0.1, 0.1)
    # AP_probs = find_current_threshold_with_noise(cell, ramp_amps, n_trials=50)
    #
    # pl.figure()
    # pl.plot(ramp_amps, AP_probs)
    # pl.xlabel('Ramp Amplitude (nA)')
    # pl.ylabel('Firing Probability')
    # pl.show()
    #
    # current_threshold = ramp_amps[np.where(AP_probs >= min_AP_prob)[0][0]]

    current_threshold = 3.5

    print 'Current threshold: %.2f nA' % current_threshold

    # simulate
    n_AP = 0
    while n_AP == 0:
        v, t, _ = simulate_rampIV(cell, current_threshold, v_init=-75)
        n_AP = len(get_AP_onset_idxs(v, threshold=0))

    cell = Cell.from_modeldir(model_dir)
    v_without_noise, t_without_noise, _ = simulate_rampIV(cell, current_threshold, v_init=-75)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'rampIV', 'with_noise', '%.2f(nA)' % current_threshold)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.savetxt(os.path.join(save_dir_img, 'current_threshold.txt'), np.array([current_threshold]))

    #plot_rampIV(t, v, save_dir_img)

    fig = pl.figure()
    pl.plot(t_without_noise, v_without_noise, 'darkred', label='no noise')
    pl.plot(t, v, 'r', label='noise')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'v.png'))
    pl.show()