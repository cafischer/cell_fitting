from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import iclamp_handling_onset, simulate_currents
import os
from cell_characteristics import to_idx
pl.style.use('paper')

__author__ = 'caro'


def get_ramp(start_idx, end_idx, amp_before, ramp_amp, amp_after):
    diff_idx = end_idx - start_idx
    half_diff_up = int(np.ceil(diff_idx / 2))
    half_diff_down = int(np.floor(diff_idx / 2))
    i_exp = np.zeros(diff_idx)
    i_exp[:half_diff_up] = np.linspace(amp_before, ramp_amp, half_diff_up)
    i_exp[half_diff_up:] = np.linspace(ramp_amp, amp_after, half_diff_down+1)[1:]
    return i_exp

def get_ramp3_times(delta_first=3, delta_ramp=2, n_times=10):
    return np.arange(delta_first, n_times * delta_ramp + delta_ramp, delta_ramp)


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

    # construct current traces
    start_ramp1 = to_idx(20, dt)
    end_ramp1 = start_ramp1 + to_idx(len_ramp, dt)
    start_step = to_idx(222, dt)
    end_step = start_step + to_idx(len_step, dt)
    start_ramp2 = end_step + to_idx(len_step2ramp, dt)
    end_ramp2 = start_ramp2 + to_idx(len_ramp, dt)

    t_exp = np.arange(0, tstop+dt, dt)
    v = np.zeros([len(ramp3_times), len(t_exp)])
    i_inj = np.zeros([len(ramp3_times), len(t_exp)])
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
        v[j], t, i_inj[j] = iclamp_handling_onset(cell, **simulation_params)

        # record currents
        currents[j], channel_list = simulate_currents(cell, simulation_params)

    return t, v, i_inj, ramp3_times, currents, channel_list, start_ramp2


def plot_double_ramp(t, v, ramp3_times, save_dir_img):
    pl.figure()
    #pl.title('1st Ramp = 4 nA, 2nd Ramp = ' + str(ramp3_amp) + ' nA')
    for j, ramp3_time in enumerate(ramp3_times):
        pl.plot(t, v[j], label='Model' if j == 0 else '', c='r')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    #pl.xlim(485, 560)
    pl.xlim(360, 400)
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'PP'+str(ramp3_amp)+'.png'))
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
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    dt = 0.01
    tstop = 500
    step_amp = 0  # 0, -0.1, 0.1
    len_step = 125
    ramp_amp = 2.9
    ramp3_times = get_ramp3_times(3, 2, 10)
    for ramp3_amp in np.arange(0, 3.55, 0.05):
        t, v, i_inj, ramp3_times, currents, channel_list, _ = double_ramp(cell, ramp_amp, ramp3_amp, ramp3_times,
                                                                       step_amp, len_step, dt, tstop)

        save_dir_img = os.path.join(save_dir, 'img', 'PP', '125', 'step' + str(step_amp))
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        plot_double_ramp(t, v, ramp3_times, save_dir_img)
        #plot_double_ramp_currents(t, v, currents, ramp3_times, channel_list, save_dir_img)
        #plot_double_ramp_currents_tmp(t, v, currents, ramp3_times, channel_list, save_dir_img)