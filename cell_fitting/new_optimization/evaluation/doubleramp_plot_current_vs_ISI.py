from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import iclamp_handling_onset
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.new_optimization.evaluation.doubleramp import get_ramp
pl.style.use('paper')

__author__ = 'caro'



def double_ramp(cell, ramp3_amp, ramp3times, step_amp, t_exp, save_dir):
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


def get_current_threshold(ramp3times):
    # simulate
    dt = 0.01
    t = np.arange(0, 800, dt)
    v_mat = np.zeros((len(seqs), len(ramp3_times), len(t)))
    ramp3_amps = np.zeros(len(seqs))
    for i, seq in enumerate(seqs):
        ramp3_amps[i] = 0 + seq * 0.05
        v_mat[i, :, :], start_ramp2 = double_ramp(cell, ramp3_amps[i], ramp3_times, step_amp, t, save_dir)

    # find current thresholds
    current_threshold = np.zeros(len(ramp3_times))
    current_threshold[:] = np.nan
    for j in range(len(ramp3_times)):
        for i in range(len(seqs)):
            onsets = get_AP_onset_idxs(v_mat[i, j, :], AP_threshold)
            onsets = onsets[onsets > start_ramp2]
            if len(onsets) > 1:  # 1st spike is mandatory, 2nd would be on the DAP
                current_threshold[j] = ramp3_amps[i]
                break
    return current_threshold, ramp3_times


if __name__ == '__main__':

    # parameters
    save_dir = '../../results/server/2017-07-27_09:18:59/22/L-BFGS-B/'
    model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../results/hand_tuning/cell_2017-07-24_13:59:54_21_0/'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    seqs = range(31)
    AP_threshold = -30
    step_amps = [-0.1, 0, 0.1]

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # simulation params
    delta_ramp = 2
    delta_first = 3
    ramp3_times = np.arange(delta_first, 10 * delta_ramp + delta_ramp, delta_ramp)

    # simulation
    current_thresholds = [0] * len(step_amps)
    for i, step_amp in enumerate(step_amps):
        current_thresholds[i], ramp3_times = get_current_threshold(ramp3_times)

    # saving
    save_dir_img = os.path.join(save_dir, 'img', 'PP')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # plot
    colors = ['b', 'k', 'r']
    pl.figure()
    for i, current_threshold in enumerate(current_thresholds):
        pl.plot(ramp3_times, current_threshold, '-o', color=colors[i], label='Step Amp.: '+str(step_amps[i]))
    pl.xlabel('$ISI_{Ramp}$ (ms)')
    pl.ylabel('Current threshold (mA/$cm^2$)')
    pl.xlim(0, ramp3_times[-1]+2)
    pl.ylim(0, 3.0)
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'current_vs_ISI.png'))
    pl.show()

# TODO: Latex same look as python standard