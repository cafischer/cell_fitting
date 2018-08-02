import os
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib.colors import to_rgb
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import extract_simulation_params, simulate_gates
from cell_fitting.util import merge_dicts
from cell_fitting.util import change_color_brightness, get_channel_dict_for_plotting, get_gate_dict_for_plotting


def plot_gates_on_ax(ax1, channel_list, gates, t, v, power_gates=None):
    new_channel_names = {k: k for k in gates.keys()}
    new_channel_names['nap_m'] = 'nat_m'
    new_channel_names['nap_h'] = 'nat_h'
    new_channel_names['nat_m'] = 'nap_m'
    new_channel_names['nat_h'] = 'nap_h'
    new_channel_names['hcn_slow_n'] = 'hcn_n'
    channel_list = ['hcn' if c == 'hcn_slow' else c for c in channel_list]

    channel_dict = get_channel_dict_for_plotting()
    gate_dict = get_gate_dict_for_plotting()

    cmap = pl.get_cmap("tab10")
    colors = {channel_name: cmap(i) for i, channel_name in enumerate(channel_list)}

    t_plot = t - 7

    ax2 = ax1.twinx()
    ax2.plot(t_plot, v, 'k')
    ax2.set_ylabel('Mem. pot. (mV)')
    ax2.spines['right'].set_visible(True)
    ax2.set_ylim(-80, -40)

    for k in sorted(new_channel_names.keys(), reverse=True):
        channel_name, gate_name = new_channel_names[k].split('_')
        color = colors[channel_name]
        if gate_name == 'm':
            color = change_color_brightness(to_rgb(color), 35, 'brighter')
        elif gate_name == 'h':
            color = change_color_brightness(to_rgb(color), 35, 'darker')

        if power_gates is None:
            ax1.plot(t_plot, gates[k],
                     label=channel_dict[channel_name]+' '+gate_dict[new_channel_names[k]], color=color)
        else:
            ax1.plot(t_plot, gates[k] ** power_gates[k],
                     label=channel_dict[channel_name]+' '+gate_dict[new_channel_names[k]], color=color)
    ax1.set_ylabel('Degree of opening')
    ax1.set_xlabel('Time (ms)')
    ax1.set_xlim(0, 55)
    ax1.legend()


if __name__ == '__main__':
    # parameters
    data_dir = '../../data/cell_csv_data/2015_08_26b/rampIV/3.1(nA).csv'
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # get simulation_params
    data = pd.read_csv(data_dir)
    sim_params = {'onset': 200, 'v_init': -75}
    simulation_params = merge_dicts(extract_simulation_params(data.v.values, data.t.values, data.i.values), sim_params)

    # plot gates
    simulate_gates(cell, simulation_params, plot=True)