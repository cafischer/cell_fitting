import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import to_rgb
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import extract_simulation_params, simulate_gates
from cell_fitting.util import merge_dicts, change_color_brightness, get_channel_dict_for_plotting, \
    get_gate_dict_for_plotting, get_channel_color_for_plotting, get_gates_of_channel
from cell_fitting.read_heka import get_i_inj_from_function
import copy


def plot_gates_on_ax(ax1, gates, t, v, power_gates=None):
    channel_dict = get_channel_dict_for_plotting()
    gate_dict = get_gate_dict_for_plotting()
    channel_color = get_channel_color_for_plotting()

    t_plot = t - 7

    ax2 = ax1.twinx()
    ax2.plot(t_plot, v, 'k', linestyle=':')
    #ax2.set_ylabel('Mem. pot. (mV)')
    #ax2.spines['right'].set_visible(True)
    ax2.set_yticks([])
    ax2.set_ylim(-80, -40)

    for k in sorted(gates.keys(), reverse=True):
        if 'hcn' in k:
            channel_name = 'hcn_slow'
            gate_name = '$h$'
        else:
            channel_name, gate_name = k.split('_')
        color = channel_color[channel_name]
        if gate_name == 'm':
            color = change_color_brightness(to_rgb(color), 35, 'brighter')
        elif gate_name == 'h':
            color = change_color_brightness(to_rgb(color), 35, 'darker')

        if power_gates is None:
            ax1.plot(t_plot, gates[k],
                     label=channel_dict[channel_name]+' '+gate_dict[k], color=color)
        else:
            ax1.plot(t_plot, gates[k] ** power_gates[k],
                     label=channel_dict[channel_name]+' '+gate_dict[k], color=color)
        nan = np.isnan(gates[k])
        if np.any(nan):
            idx_first_nan = np.where(nan)[0][0]
            ax1.plot(np.insert(t_plot[nan], 0, t_plot[idx_first_nan-1]),
                     np.insert(np.zeros(sum(nan)), 0, gates[k][idx_first_nan-1]),
                     linestyle='--', color=color)
    ax1.set_ylabel('Degree of opening')
    ax1.set_xlabel('Time (ms)')
    ax1.set_xlim(0, 55)
    ax1.legend()


def plot_product_gates_on_ax(ax1, channel_list, gates, t, v, power_gates):
    channel_list = copy.copy(channel_list)
    channel_list.remove('pas')
    channel_dict = get_channel_dict_for_plotting()
    channel_color = get_channel_color_for_plotting()
    gates_of_channel = get_gates_of_channel()
    channel_gate_product = {'nat': '$\mathrm{m^3h}$', 'nap': '$\mathrm{m^3h}$', 'kdr': '$\mathrm{m^4}$',
                            'hcn_slow': '$\mathrm{h}$'}

    t_plot = t - 7

    ax2 = ax1.twinx()
    ax2.plot(t_plot, v, 'k', linestyle=':')
    # ax2.set_ylabel('Mem. pot. (mV)')
    # ax2.spines['right'].set_visible(True)
    ax2.set_yticks([])
    ax2.set_ylim(-80, -40)

    for i, channel_name in enumerate(sorted(channel_list, reverse=True)):
        gate_names = gates_of_channel[channel_name]
        color = channel_color[channel_name]
        channel_gates = [channel_name+'_'+gate_name for gate_name in gate_names]
        gate_powers = np.array([gates[k] ** power_gates[k] for k in channel_gates])
        ax1.plot(t_plot, np.product(gate_powers, axis=0),
                 label=channel_dict[channel_name] + ' ' + channel_gate_product[channel_name], color=color)
    ax1.set_ylabel('Degree of opening')
    ax1.set_xlabel('Time (ms)')
    ax1.set_xlim(0, 55)
    ax1.legend()


if __name__ == '__main__':
    # parameters
    data_dir = '../../data/cell_csv_data/2015_08_26b/rampIV/3.1(nA).csv'
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # get simulation_params
    # data = pd.read_csv(data_dir)
    # sim_params = {'onset': 200, 'v_init': -75}
    # simulation_params = merge_dicts(extract_simulation_params(data.v.values, data.t.values, data.i.values), sim_params)
    i_exp = get_i_inj_from_function('IV', [0], 1000, 0.01)[0]
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -75, 'tstop': 1000,
                         'dt': 0.01, 'celsius': 35, 'onset': 200}

    # plot gates
    simulate_gates(cell, simulation_params, plot=True)