from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import iclamp_handling_onset
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs, to_idx
from cell_fitting.new_optimization.evaluation.doubleramp import get_ramp
pl.style.use('paper')

__author__ = 'caro'



def double_ramp(cell, ramp3_amp, ramp3_times, step_amp, t_exp):
    """
    original values
    delta_ramp = 2
    delta_first = 3
    ramp3_times = np.arange(delta_first, 10 * delta_ramp + delta_ramp, delta_ramp)
    baseline_amp = -0.05
    ramp_amp = 4.0
    ramp3_amp = 1.8
    step_amp = 0  # or -0.1 or 0.1
    dt = 0.01

    amplitude of second ramp goes up by 0.05 nA after each sequence
    """

    baseline_amp = -0.05
    ramp_amp = 4.0
    dt = t_exp[1] - t_exp[0]

    # construct current traces
    len_ramp = 3
    start_ramp1 = int(round(20 / dt))
    end_ramp1 = start_ramp1 + int(round(len_ramp / dt))
    start_step = int(round(222 / dt))
    end_step = start_step + int(round(250 / dt))
    start_ramp2 = end_step + int(round(15 / dt))
    end_ramp2 = start_ramp2 + int(round(len_ramp / dt))

    vs = np.zeros([len(ramp3_times), len(t_exp)])
    currents = [0] * len(ramp3_times)

    for j, ramp3_time in enumerate(ramp3_times):
        start_ramp3 = start_ramp2 + int(round(ramp3_time / dt))
        end_ramp3 = start_ramp3 + int(round(len_ramp / dt))

        i_exp = np.ones(len(t_exp)) * baseline_amp
        i_exp[start_ramp1:end_ramp1] = get_ramp(start_ramp1, end_ramp1, 0, ramp_amp, 0)
        i_exp[start_step:end_step] = step_amp
        i_exp[start_ramp2:end_ramp2] = get_ramp(start_ramp2, end_ramp2, 0, ramp_amp, 0)
        i_exp[start_ramp3:end_ramp3] = get_ramp(start_ramp3, end_ramp3, 0, ramp3_amp, 0)

        # get simulation parameters
        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -75, 'tstop': t_exp[-1],
                             'dt': dt, 'celsius': 35, 'onset': 300}

        # record v
        vs[j], t, _ = iclamp_handling_onset(cell, **simulation_params)

    return vs, start_ramp2


def simulate_and_get_current_threshold():

    dt = 0.01
    t = np.arange(0, 800, dt)
    ramp3_amps = np.arange(0, 4.55, 0.05)
    delta_ramp = 2
    delta_first = 3
    ramp3_times = np.arange(delta_first, 10 * delta_ramp + delta_ramp, delta_ramp)

    v_mat = np.zeros((len(ramp3_amps), len(ramp3_times), len(t)))
    for i, seq in enumerate(ramp3_amps):
        v_mat[i, :, :], start_ramp2 = double_ramp(cell, ramp3_amps[i], ramp3_times, step_amp, t)

    dt = t[1] - t[0]
    start_ramp1 = to_idx(20, dt)
    v_dap = v_mat[0, 0, start_ramp1:start_ramp1 + to_idx(ramp3_times[-1] + ramp3_times[0], dt)]
    t_dap = np.arange(len(v_dap)) * dt

    current_thresholds = get_current_threshold(v_mat, ramp3_amps, ramp3_times, start_ramp2, AP_threshold)

    return current_thresholds, ramp3_times, ramp3_amps, v_dap, t_dap


def get_current_threshold(v_mat, ramp3_amps, ramp3_times, start_ramp2, AP_threshold=None):

    current_thresholds = np.zeros(len(ramp3_times))
    current_thresholds[:] = np.nan

    for j in range(len(ramp3_times)):  # order of for loops important (find lowest amp that produces spike)
        for i in range(len(ramp3_amps)):
            onsets = get_AP_onset_idxs(v_mat[i, j, :], AP_threshold)
            onsets = onsets[onsets > start_ramp2]
            if len(onsets) > 1:  # 1st spike is mandatory, 2nd would be on the DAP
                current_thresholds[j] = ramp3_amps[i]
                break
    return current_thresholds


def plot_current_threshold(current_thresholds, ramp3_times, step_amps, ramp3_amp_min, ramp3_amp_max, v_dap, t_dap,
                           save_dir_img, legend_loc='upper left'):

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    colors_dict = {-0.1: 'b', 0.0: 'k', 0.1: 'r'}
    colors = [colors_dict[amp] for amp in step_amps]

    fig, ax = pl.subplots()

    # plot v_dap
    ax2 = ax.twinx()
    ax2.plot(t_dap, v_dap, 'k')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.spines['right'].set_visible(True)

    # plot current threshold
    ax.axhline(ramp3_amp_min, linestyle='--', c='0.5')
    ax.axhline(ramp3_amp_max, linestyle='--', c='0.5')
    for i, current_threshold in enumerate(current_thresholds):
        ax.plot(ramp3_times, current_threshold, '-o', color=colors[i], label='Step Amp.: '+str(step_amps[i]),
                markersize=9-2.5*i)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current threshold (mA/$cm^2$)')
    ax.set_xticks(ramp3_times)
    ax.set_xlim(0, ramp3_times[-1]+2)
    ax.set_ylim(0, 4.5)
    ax.legend(loc=legend_loc)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'current_threshold.png'))
    #pl.show()

    fig, ax = pl.subplots()

    # plot v_dap
    ax2 = ax.twinx()
    ax2.plot(t_dap, v_dap, 'k')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.spines['right'].set_visible(True)

    # plot current threshold
    ax.axhline(ramp3_amp_min, linestyle='--', c='0.5')
    ax.axhline(ramp3_amp_max, linestyle='--', c='0.5')
    for i, current_threshold in enumerate(current_thresholds):
        ax.plot(ramp3_times, current_threshold, '-o', color=colors[i], label='Step Amp.: '+str(step_amps[i]),
                markersize=9-2.5*i)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current threshold (mA/$cm^2$)')
    ax.set_xticks(ramp3_times)
    ax.set_xlim(0, ramp3_times[-1]+2)
    ax.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'current_threshold_zoom.png'))
    pl.show()


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    model_dir = os.path.join(save_dir, 'cell.json')
    #save_dir = '../../results/server/2017-07-27_09:18:59/22/L-BFGS-B/'
    #model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../results/hand_tuning/cell_2017-07-24_13:59:54_21_0/'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    AP_threshold = -30
    step_amps = [-0.1, 0, 0.1]

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # simulation
    current_thresholds = [0] * len(step_amps)
    for i, step_amp in enumerate(step_amps):
        current_thresholds[i], ramp3_times, ramp3_amps, v_dap, t_dap = simulate_and_get_current_threshold()

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'PP')
    plot_current_threshold(current_thresholds, ramp3_times, step_amps, ramp3_amps[0], ramp3_amps[-1], v_dap, t_dap,
                           save_dir_img)