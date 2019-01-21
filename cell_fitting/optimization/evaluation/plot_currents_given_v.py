import copy
import os
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.simulate import currents_given_v, iclamp_handling_onset
from cell_fitting.util import get_channel_dict_for_plotting, get_channel_color_for_plotting
from cell_fitting.read_heka import get_sweep_index_for_amp, get_v_and_t_from_heka, get_i_inj_from_function
from cell_fitting.optimization.helpers import get_channel_list, get_ionlist
pl.style.use('paper')


def plot_currents_on_ax(ax1, channel_list, currents, t, v):
    channel_dict = get_channel_dict_for_plotting()
    channel_color = get_channel_color_for_plotting()

    t_plot = t - 7

    ax2 = ax1.twinx()
    ax2.plot(t_plot, v, 'k', linestyle=':')
    #ax2.set_ylabel('Mem. pot. (mV)')
    #ax2.spines['right'].set_visible(True)
    ax2.set_yticks([])

    for i in range(len(channel_list)):
        ax1.plot(t_plot, -1 * currents[i], color=channel_color[channel_list[i]],
                 label=channel_dict[channel_list[i]])
        nan = np.isnan(currents[i])
        if np.any(nan):
            idx_first_nan = np.where(nan)[0][0]
            ax1.plot(np.insert(t_plot[nan], 0, t_plot[idx_first_nan-1]),
                     np.insert(np.zeros(sum(nan)), 0, -1 * currents[i][idx_first_nan-1]),
                     linestyle='--', color=channel_color[channel_list[i]])
        ax1.set_ylabel('Current (mA/cm$^2$)')
        ax1.set_xlabel('Time (ms)')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
    ax2.set_ylim(-80, -40)
    ax1.set_ylim(-0.1, 0.1)
    ax1.set_xlim(0, 55)


if __name__ == '__main__':
    # parameters
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # get simulation_params
    # sweep_idx = get_sweep_index_for_amp(3.1, 'rampIV')
    # v_exp, t_exp = get_v_and_t_from_heka(data_dir, 'rampIV', sweep_idxs=[sweep_idx])
    # v_exp = v_exp[0]
    # t_exp = t_exp[0]

    i_exp = get_i_inj_from_function('rampIV', [get_sweep_index_for_amp(3.1, 'rampIV')], 150, 0.01)[0]
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -75, 'tstop': 1000,
                         'dt': 0.01, 'celsius': 35, 'onset': 200}
    v_exp, t_exp, _ = iclamp_handling_onset(cell, **simulation_params)

    # TODO
    mechanism_dir2 = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/resurgent'
    load_mechanism_dir(mechanism_dir2)
    cell.insert_mechanisms([[['soma', '0.5', 'narsg', 'gbar']]])
    cell.update_attr(['soma', '0.5', 'narsg', 'gbar'], 0.1)

    # plot currents
    channel_list = get_channel_list(cell, 'soma')
    currents = currents_given_v(v_exp, t_exp, cell.soma, channel_list, get_ionlist(channel_list), 35, plot=False)

    fig, ax1 = pl.subplots()
    for i in range(len(channel_list)):
        ax1.plot(t_exp, -1 * currents[i], label=channel_list[i])
        ax1.set_ylabel('Current (mA/cm$^2$)')
        ax1.set_xlabel('Time (ms)')
    ax2 = ax1.twinx()
    ax2.plot(t_exp, v_exp, 'k', label='Mem. Pot.')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.spines['right'].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
    #ax2.set_ylim(-80, -40)
    ax1.set_ylim(-0.1, 0.1)
    pl.tight_layout()
    pl.show()

    fig, ax1 = pl.subplots()
    for i in range(len(channel_list)):
        ax1.plot(t_exp, -1 * currents[i], label=channel_list[i])
        ax1.set_ylabel('Current (mA/cm$^2$)')
        ax1.set_xlabel('Time (ms)')
    ax2 = ax1.twinx()
    ax2.plot(t_exp, v_exp, 'k', label='Mem. Pot.')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.spines['right'].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
    # ax2.set_ylim(-80, -40)
    ax1.set_ylim(-0.1, 0.1)
    pl.tight_layout()
    pl.show()