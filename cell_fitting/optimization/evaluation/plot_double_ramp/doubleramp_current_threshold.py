from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
from cell_characteristics.analyze_APs import get_AP_onset_idxs, to_idx
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.plot_double_ramp.plot_doubleramp import double_ramp, get_ramp3_times
from cell_fitting.util import init_nan

pl.style.use('paper')

__author__ = 'caro'


def simulate_and_get_current_threshold(cell, step_amp):

    dt = 0.01
    tstop = 500
    t = np.arange(0, tstop+dt, dt)
    ramp_amp = 4.0
    ramp3_amps = np.arange(0, 4.0+0.05, 0.05)
    ramp3_times = get_ramp3_times(3, 2, 10)
    len_step = 125
    AP_threshold = -10

    v_mat = np.zeros((len(ramp3_amps), len(ramp3_times), len(t)))
    for i, seq in enumerate(ramp3_amps):
        t, v_mat[i, :, :], i_inj, ramp3_times, _, _, start_ramp2 = double_ramp(cell, ramp_amp, ramp3_amps[i],
                                                                               ramp3_times, step_amp, len_step,
                                                                               dt, tstop)

    start_ramp1 = to_idx(20, dt)
    v_dap = v_mat[0, 0, start_ramp1:start_ramp1 + to_idx(ramp3_times[-1] + ramp3_times[0], dt)]
    t_dap = np.arange(len(v_dap)) * dt

    current_thresholds = get_current_threshold(v_mat, ramp3_amps, ramp3_times, start_ramp2, dt, AP_threshold)

    return current_thresholds, ramp3_times, ramp3_amps, v_dap, t_dap, v_mat, t


def get_current_threshold(v_mat, ramp3_amps, ramp3_times, start_ramp2_idx, dt, AP_threshold=None):

    current_thresholds = init_nan(len(ramp3_times))
    #pl.figure()
    for j in range(len(ramp3_times)):  # order of for loops important (find lowest amp that produces spike)
        #pl.plot(v_mat[0, j, :], label=str(ramp3_times[j]))
        for i in range(len(ramp3_amps)):
            onsets = get_AP_onset_idxs(v_mat[i, j, :], AP_threshold)
            onsets = onsets[onsets > start_ramp2_idx]
            if len(onsets) > 1 and onsets[1] - onsets[0] <= to_idx(3, dt):  #  sometimes 1st AP gets 2 onsets because charging is so high
                onsets = onsets[1:]
            if len(onsets) > 1:  # 1st spike is mandatory, 2nd would be on the DAP
                    current_thresholds[j] = ramp3_amps[i]
                    break
    #pl.legend()
    #pl.show()
    return current_thresholds


def plot_current_threshold(current_thresholds, current_threshold_rampIV, ramp3_times, step_amps, ramp3_amp_min,
                           ramp3_amp_max, v_dap, t_dap, save_dir_img=None, legend_loc='upper left'):

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    colors_dict = {-0.1: 'b', 0.0: 'k', 0.1: 'r'}
    colors = [colors_dict[amp] for amp in step_amps]

    # plot current threshold
    fig, ax = pl.subplots()
    ax2 = ax.twinx()
    ax2.plot(t_dap, v_dap, 'k')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.spines['right'].set_visible(True)

    ax.axhline(ramp3_amp_min, linestyle='--', c='0.5')
    ax.axhline(ramp3_amp_max, linestyle='--', c='0.5')
    ax.plot(0, current_threshold_rampIV, 'ok', markersize=6.5)
    for i, current_threshold in enumerate(current_thresholds):
        ax.plot(ramp3_times, current_threshold, '-o', color=colors[i], label='Step Amp.: ' + str(step_amps[i]),
                markersize=9 - 2.5 * i)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current threshold (mA/$cm^2$)')
    ax.set_xticks(np.insert(ramp3_times, 0, [0]))
    ax.set_xlim(-0.5, ramp3_times[-1] + 2)
    ax.set_ylim(0, 4.2)
    ax.legend(loc=legend_loc)
    pl.tight_layout()
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'current_threshold.png'))
    return fig



def save_diff_current_threshold(current_threshold_rampIV, current_thresholds, save_dir):
    diff_current_threshold = current_threshold_rampIV - np.nanmin(current_thresholds[1])
    np.savetxt(save_dir, np.array([diff_current_threshold]))


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    AP_threshold = 0
    step_amps = [-0.1, 0, 0.1]

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # load current threshold
    current_threshold_rampIV = float(np.loadtxt(os.path.join(save_dir, 'img', 'rampIV', 'current_threshold.txt')))

    # simulation
    current_thresholds = [0] * len(step_amps)
    for i, step_amp in enumerate(step_amps):
        current_thresholds[i], ramp3_times, ramp3_amps, v_dap, t_dap, _ = simulate_and_get_current_threshold(cell)

    # save difference of minimal current threshold and from rampIV
    save_diff_current_threshold(current_threshold_rampIV, current_thresholds,
                                os.path.join(save_dir, 'img', 'rampIV', 'diff_current_threshold.txt'))

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'PP', '125')
    plot_current_threshold(current_thresholds, current_threshold_rampIV, ramp3_times, step_amps, ramp3_amps[0],
                           ramp3_amps[-1], v_dap, t_dap, save_dir_img)