from __future__ import division
import os
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
from cell_characteristics.analyze_APs import get_AP_onset_idxs, to_idx
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.evaluation.plot_double_ramp.plot_doubleramp import double_ramp, get_ramp3_times
from cell_fitting.util import init_nan
from cell_fitting.investigate_grid_cell_stimuli.model_noise.with_OU import ou_noise_input
pl.style.use('paper')

__author__ = 'caro'


def simulate_and_get_current_threshold(cell, step_amp):

    dt = 0.01
    tstop = 500
    t = np.arange(0, tstop+dt, dt)
    ramp_amp = 4.0
    ramp3_amps = np.arange(1.5, 3.0+0.05, 0.05)  # TODO np.arange(0, 4.0+0.05, 0.05)
    ramp3_times = get_ramp3_times(3, 2, 5)   # TODO: get_ramp3_times(3, 2, 10)
    len_step = 125
    AP_threshold = -10

    v_mat = np.zeros((len(ramp3_amps), len(ramp3_times), len(t)))
    for i, seq in enumerate(ramp3_amps):
        t, v_mat[i, :, :], i_inj, ramp3_times, _, _, start_ramp2_idx = double_ramp(cell, ramp_amp, ramp3_amps[i],
                                                                               ramp3_times, step_amp, len_step,
                                                                               dt, tstop)

    start_ramp1 = to_idx(20, dt)
    v_dap = v_mat[0, 0, start_ramp1:start_ramp1 + to_idx(ramp3_times[-1] + ramp3_times[0], dt)]
    t_dap = np.arange(len(v_dap)) * dt

    current_thresholds = get_current_threshold(v_mat, ramp3_amps, ramp3_times, start_ramp2_idx, dt, AP_threshold)

    return current_thresholds, ramp3_times, ramp3_amps, v_dap, t_dap, v_mat, t, start_ramp2_idx


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


def count_APs_per_ramp_amp(v_mat, ramp3_amps, ramp3_times, start_ramp2_idx, dt, AP_threshold=None):

    count_APs = np.zeros((len(ramp3_amps), len(ramp3_times)))
    for j in range(len(ramp3_times)):  # order of for loops important (find lowest amp that produces spike)
        for i in range(len(ramp3_amps)):
            onsets = get_AP_onset_idxs(v_mat[i, j, :], AP_threshold)
            onsets = onsets[onsets > start_ramp2_idx]
            if (len(onsets) > 1  # 1st spike is mandatory and 2nd should be on the DAP
                    and start_ramp2_idx + to_idx(ramp3_times[j], dt) < onsets[1] < start_ramp2_idx + to_idx(ramp3_times[j]+3, dt)):
                count_APs[i, j] = 1
                # pl.figure()
                # pl.plot(np.arange(len(v_mat[i, j, start_ramp2_idx:]))*dt, v_mat[i, j, start_ramp2_idx:])
                # pl.show()
    return count_APs


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


def plot_double_ramp(v_mat, t, ramp3_times, ramp3_amps, save_dir_img):
    for i, ramp3_amp in enumerate(ramp3_amps):
        pl.figure()
        for j, ramp3_time in enumerate(ramp3_times):
            pl.plot(t, v_mat[i, j, :], c='r')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        #pl.xlim(485, 560)
        pl.xlim(360, 400)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'PP %.2f' % ramp3_amp+'.png'))
        #pl.show()


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    AP_threshold = 0
    step_amps =  [-0.1, 0] # [-0.1, 0, 0.1]
    save_dir_img = os.path.join(save_dir, 'img', 'PP', 'with_noise_test', '125')
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")
    #noise_params = {'g_e0': 0.025, 'g_i0': 0.08, 'std_e': 0.004, 'std_i': 0.006, 'tau_e': 2.0, 'tau_i': 5.0}
    noise_params = {'g_e0': 0.0001, 'g_i0': 0.05, 'std_e': 0.007, 'std_i': 0.006, 'tau_e': 2.4, 'tau_i': 5.0}
    n_trials = 2

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    ou_process = ou_noise_input(cell, **noise_params)

    # load current threshold
    current_threshold_rampIV = float(np.loadtxt(os.path.join(save_dir, 'img', 'rampIV', 'current_threshold.txt')))

    current_thresholds = []
    current_thresholds_3d = [0] * 3
    for step_amp_idx, step_amp in enumerate(step_amps):
        count_APs_trials = []
        for trial in range(n_trials):
            ou_process.new_seed(trial)

            # simulation
            _, ramp3_times, ramp3_amps, v_dap, t_dap, v_mat, t, start_ramp2_idx = simulate_and_get_current_threshold(cell, step_amp)

            count_APs = count_APs_per_ramp_amp(v_mat, ramp3_amps, ramp3_times, start_ramp2_idx, t[1]-t[0], AP_threshold)
            count_APs_trials.append(count_APs)

            save_dir_img_step = os.path.join(save_dir_img, 'step'+str(step_amp), 'trial'+str(trial))
            if not os.path.exists(save_dir_img_step):
                os.makedirs(save_dir_img_step)
            plot_double_ramp(v_mat, t, ramp3_times, ramp3_amps, save_dir_img_step)
            pl.close()

        current_threshold = init_nan(len(ramp3_times))
        current_thresholds_3d[step_amp_idx] = np.zeros((len(ramp3_amps), len(ramp3_times)))
        for j in range(len(ramp3_times)):
            count_APs_trials_tmp = np.vstack([count_APs_trials[tr][:, j] for tr in range(n_trials)]).T
            mean_count_APs_trials_tmp = np.mean(count_APs_trials_tmp, 1)
            max_APs_idx = np.argmax(mean_count_APs_trials_tmp)
            current_thresholds_3d[step_amp_idx][:, j] = mean_count_APs_trials_tmp
            if not mean_count_APs_trials_tmp[max_APs_idx] == 0:
                current_threshold[j] = ramp3_amps[max_APs_idx]

        current_thresholds.append(current_threshold)

    # plot
    with open(os.path.join(save_dir_img, 'noise_params.json')) as f:
        json.dump(noise_params, f)

    plot_current_threshold(current_thresholds, current_threshold_rampIV, ramp3_times, step_amps, ramp3_amps[0],
                           ramp3_amps[-1], v_dap, t_dap, save_dir_img, legend_loc='lower right')


    # plot current threshold 3d
    colors_dict = {-0.1: 'b', 0.0: 'k', 0.1: 'r'}
    colors = [colors_dict[amp] for amp in step_amps]

    fig = pl.figure()
    ax = fig.gca(projection='3d')
    for i, step_amp in enumerate(step_amps):
        for ramp3_time_idx, ramp3_time in enumerate(ramp3_times):
            ax.plot(np.ones(len(ramp3_amps)) * ramp3_time, ramp3_amps, current_thresholds_3d[i][:, ramp3_time_idx],
                    linestyle='-', color=colors[i], markersize=9 - 2.5 * i,
                    label='Step Amp.: ' + str(step_amps[i]))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current Amplitude (nA)')
    ax.set_zlabel('Firing Probability')
    ax.set_xticks(np.insert(ramp3_times, 0, [0]))
    ax.set_xlim(-0.5, ramp3_times[-1] + 2)
    ax.set_ylim(0, 4.2)
    ax.view_init(30, -135)
    # ax.legend(loc='lower right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'current_threshold_3d.png'))
    #pl.show()