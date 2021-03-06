from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
from cell_characteristics import to_idx
from nrn_wrapper import Cell
from cell_fitting.read_heka.i_inj_functions import get_i_inj_double_ramp
from cell_fitting.optimization.simulate import iclamp_handling_onset, simulate_currents
from cell_fitting.data.plot_doubleramp import get_ramp3_times
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.util import merge_dicts
pl.style.use('paper')
__author__ = 'caro'


def double_ramp(cell, ramp_amp, ramp3_amp, ramp3_times, step_amp, len_step, dt, tstop):
    """
    original values
    delta_ramp = 2
    delta_first = 3
    ramp3_times = np.arange(delta_first, 10 * delta_ramp + delta_ramp, delta_ramp)
    baseline_amp = -0.05
    ramp_amp = 4.0
    ramp3_amp = 1.8
    step_amp = 0  # or -0.1 or 0.1
    dur_step = 250
    dt = 0.01
    len_step2ramp = 15
    tstop = 800
    len_ramp = 3

    amplitude of second ramp goes up by 0.05 nA after each sequence
    """

    baseline_amp = -0.05
    len_step2ramp = 15
    len_ramp = 2
    # start_ramp1 = to_idx(20, dt)
    # end_ramp1 = start_ramp1 + to_idx(len_ramp, dt)
    start_step = to_idx(222, dt)
    end_step = start_step + to_idx(len_step, dt)
    start_ramp2 = end_step + to_idx(len_step2ramp, dt)
    # end_ramp2 = start_ramp2 + to_idx(len_ramp, dt)

    t_exp = np.arange(0, tstop+dt, dt)
    v_mat = np.zeros([len(ramp3_times), len(t_exp)])
    i_inj_mat = np.zeros([len(ramp3_times), len(t_exp)])
    currents = [0] * len(ramp3_times)

    for j, ramp3_time in enumerate(ramp3_times):
        i_exp = get_i_inj_double_ramp(ramp_amp, ramp3_amp, ramp3_time, step_amp, len_step, baseline_amp, len_ramp,
                                      len_step2ramp=len_step2ramp, tstop=tstop, dt=dt)

        # get simulation parameters
        simulation_params = merge_dicts(get_standard_simulation_params(),
                                        {'i_inj': i_exp, 'tstop': t_exp[-1], 'dt': dt})

        # record v
        v_mat[j], t, i_inj_mat[j] = iclamp_handling_onset(cell, **simulation_params)

        # record currents
        currents[j], channel_list = simulate_currents(cell, simulation_params)

    return t, v_mat, i_inj_mat, ramp3_times, currents, channel_list, start_ramp2


def plot_double_ramp(t, v_mat, ramp3_times, save_dir_img):
    pl.figure()
    for j, ramp3_time in enumerate(ramp3_times):
        pl.plot(t, v_mat[j], c='r')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    #pl.xlim(485, 560)
    pl.xlim(360, 400)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'PP'+str(ramp3_amp)+'.png'))
    pl.show()


def plot_double_ramp_explanation_img(t, v_mats, i_inj_mats, ramp3_times, save_dir_img):
    pl.figure()
    for i in range(1, len(ramp3_times)):
        pl.plot(t, i_inj_mats[0][i], '--k', linewidth=0.8)
    pl.plot(t, i_inj_mats[1][0], c='r', linewidth=1.0)
    pl.plot(t, i_inj_mats[2][0], c='b', linewidth=1.0)
    pl.plot(t, i_inj_mats[0][0], c='k', linewidth=1.0)
    pl.xlim(340, 400)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'i_inj_zoom.png'))

    pl.figure()
    for i in range(1, len(ramp3_times)):
        pl.plot(t, i_inj_mats[0][i], '--k', linewidth=0.8)
    pl.plot(t, i_inj_mats[1][0], c='r', linewidth=1.0)
    pl.plot(t, i_inj_mats[2][0], c='b', linewidth=1.0)
    pl.plot(t, i_inj_mats[0][0], c='k', linewidth=1.0)
    pl.xlabel('Time (ms)')
    pl.ylabel('Current (nA)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'i_inj.png'))
    #pl.show()

    pl.figure()
    for i in range(1, len(ramp3_times)):
        pl.plot(t, v_mats[0][i], '--k', linewidth=0.8)
    pl.plot(t, v_mats[1][0], c='r', linewidth=1.0)
    pl.plot(t, v_mats[2][0], c='b', linewidth=1.0)
    pl.plot(t, v_mats[0][0], c='k', linewidth=1.0)
    pl.xlabel('Time (ms)')
    pl.ylabel('Current (nA)')
    pl.xlim(340, 400)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'v_zoom.png'))

    pl.figure()
    for i in range(1, len(ramp3_times)):
        pl.plot(t, v_mats[0][i], '--k', linewidth=0.8)
    pl.plot(t, v_mats[1][0], c='r', linewidth=1.0)
    pl.plot(t, v_mats[2][0], c='b', linewidth=1.0)
    pl.plot(t, v_mats[0][0], c='k', linewidth=1.0)
    pl.xlabel('Time (ms)')
    pl.ylabel('Current (nA)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'v.png'))
    #pl.show()


def plot_double_ramp_currents(t, v, currents, ramp3_times, channel_list, save_dir_img):
    fig, ax1 = pl.subplots()
    #ax1.set_title('1st Ramp = 4 nA, 2nd Ramp = ' + str(ramp3_amp) + ' nA')
    ax2 = ax1.twinx()
    colors = pl.cm.plasma(np.linspace(0, 1, len(currents[0])))
    for j, ramp3_time in enumerate(ramp3_times):
        ax2.plot(t, v[j], 'k', label='Mem. Pot.' if j == 0 else '')
        for i, current in enumerate(currents[j]):
            ax1.plot(t, -1*current, label=channel_list[i] if j == 0 else '', c=colors[i])
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current (mA/cm$^2$)')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.spines['right'].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
    pl.xlim(485, 560)
    pl.tight_layout(360, 400)
    #pl.savefig(os.path.join(save_dir_img, 'PP_currents'+str(ramp3_amp)+'.png'))
    pl.show()


def plot_double_ramp_currents_tmp(t, v, currents, ramp3_times, channel_list, save_dir_img):
    idx_ramp2 = np.argmin(np.abs(t - 493.5))
    idx_ramp3 = np.argmin(np.abs(t - 495.5))

    fig, ax1 = pl.subplots()
    colors = pl.cm.plasma(np.linspace(0, 1, len(currents[0])))
    for i in range(len(currents[0])):
        ax1.plot(0, -1 * currents[1][i][idx_ramp2] - -1 * currents[2][i][idx_ramp3],
                 label=channel_list[i], c=colors[i], marker='o')
    ax1.set_ylabel('$Current_{Ramp2} - Current_{Ramp3}$ (mA/cm$^2$)')
    ax1.legend()
    pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'PP_currents'+str(ramp3_amp)+'.png'))
    pl.show()

    fig, ax1 = pl.subplots()
    ax2 = ax1.twinx()
    colors = pl.cm.plasma(np.linspace(0, 1, len(currents[0])))
    for j in [1, 2]:  # plot ramp 1 and 2
        ax2.plot(t, v[j], 'k', label='Mem. Pot.' if j == 1 else '')
    for i in range(len(currents[0])):
        ax1.plot([t[idx_ramp2], t[idx_ramp3]], [-1 * currents[1][i][idx_ramp2], -1 * currents[2][i][idx_ramp3]],
                 label=channel_list[i], c=colors[i])
    ax1.set_ylim(-0.3, 1.0)
    pl.xlim(485, 560)
    ax1.axvline(493.5, c='0.5')
    ax1.axvline(495.5, c='0.5')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current (mA/cm$^2$)')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.spines['right'].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
    pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'PP_currents'+str(ramp3_amp)+'.png'))
    pl.show()

    fig, ax1 = pl.subplots()
    ax2 = ax1.twinx()
    colors = pl.cm.plasma(np.linspace(0, 1, len(currents[0])))
    for j in [1, 2]:  # plot ramp 1 and 2
        ax2.plot(t, v[j], 'k', label='Mem. Pot.' if j == 1 else '')
        for i, current in enumerate(currents[j]):
            ax1.plot(t, -1 * current, label=channel_list[i] if j == 1 else '', c=colors[i])
    ax1.axvline(493.5, c='0.5')
    ax1.axvline(495.5, c='0.5')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current (mA/cm$^2$)')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.spines['right'].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
    pl.xlim(485, 560)
    pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'PP_currents'+str(ramp3_amp)+'.png'))
    pl.show()


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    model_dir = os.path.join(save_dir, 'cell_rounded.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    step_idx_dict = {-0.1: 0, 0: 1, 0.1: 2}

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    dt = 0.01
    tstop = 691.99  #500
    len_step = 250  #125
    ramp_amp = 4.0
    ramp3_times = get_ramp3_times(3, 2, 10)
    ramp3_amps = np.arange(1.0, 4.0+0.05, 0.05)
    step_amps = [0, 0.1, -0.1]

    v_mat = np.zeros((len(ramp3_amps), len(ramp3_times), int(tstop/dt+1), len(step_amps)))
    for step_amp in step_amps:
        step_str = 'step_%.1f(nA)' % step_amp
        for ramp3_amp_idx, ramp3_amp in enumerate(ramp3_amps):
            t, v_mat[ramp3_amp_idx, :, :, step_idx_dict[step_amp]], i_inj_mat, \
            ramp3_times, currents, channel_list, _ = double_ramp(cell, ramp_amp, ramp3_amp, ramp3_times,
                                                                 step_amp, len_step, dt, tstop)

            # save_dir_img = os.path.join(save_dir, 'img', 'PP', str(len_step), step_str)
            # if not os.path.exists(save_dir_img):
            #     os.makedirs(save_dir_img)
            #plot_double_ramp(t, v_mat[i, :, :], ramp3_times, save_dir_img)
            #plot_double_ramp_currents(t, v, currents, ramp3_times, channel_list, save_dir_img)
            #plot_double_ramp_currents_tmp(t, v, currents, ramp3_times, channel_list, save_dir_img)

    # save
    save_dir_model = os.path.join(save_dir, 'img', 'PP', str(len_step))
    if not os.path.exists(save_dir_model):
        os.makedirs(save_dir_model)
    np.save(os.path.join(save_dir_model, 'v_mat.npy'), v_mat)