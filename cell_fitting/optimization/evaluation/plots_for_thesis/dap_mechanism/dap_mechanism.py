import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb
from cell_fitting.util import change_color_brightness
from nrn_wrapper import Cell
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model, simulate_model_currents, simulate_model_gates
from cell_fitting.optimization.evaluation.plot_currents import plot_currents_on_ax
from cell_fitting.optimization.evaluation.plot_gates import plot_gates_on_ax, plot_product_gates_on_ax
from cell_fitting.optimization.evaluation.plot_blocking.block_channel import block_channel, plot_channel_block_on_ax
from cell_fitting.test_channels.channel_characteristics import boltzmann_fun
from cell_fitting.util import get_channel_dict_for_plotting, get_channel_color_for_plotting
from cell_fitting.sensitivity_analysis.plot_correlation_param_characteristic import plot_corr_on_ax
pl.style.use('paper_subplots')


def plot_act_inact_on_ax(ax, v_range, curve_act, curve_inact, channel_name):
    channel_dict = get_channel_dict_for_plotting()
    channel_color = get_channel_color_for_plotting()
    ax.fill_between(v_range, 0, [min(c_act, c_inact) for c_act, c_inact in zip(curve_act, curve_inact)], color='0.9')
    ax.plot(v_range, curve_act, color=change_color_brightness(to_rgb(channel_color[channel_name]), 35, 'brighter'),
             label=channel_dict['nat']+' Act.')
    ax.plot(v_range, curve_inact, color=change_color_brightness(to_rgb(channel_color[channel_name]), 35, 'darker'),
             label=channel_dict['nat']+' Inact.')
    ax.set_xlabel('Mem. pot. (mV)')
    ax.set_ylabel('Degree of opening')
    ax.legend()


# TODO: colors
# TODO: check simulation_params (e.g. dt)
# TODO: maybe add blocking Nat and reconstructing AP
if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    v_init = -75

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'), mechanism_dir)

    # plot
    fig = pl.figure(figsize=(11, 7))
    outer = gridspec.GridSpec(2, 3)

    # simulate for ionic currents and gates
    ramp_amp = 3.5
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell+'.dat'), 'rampIV', ramp_amp)
    v_data += - v_data[0] + v_init
    v_model, t_model, _ = simulate_model(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init)
    currents, channel_list = simulate_model_currents(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init)
    gates, power_gates = simulate_model_gates(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init)

    # ionic currents
    ax = pl.Subplot(fig, outer[0, 0])
    fig.add_subplot(ax)

    plot_currents_on_ax(ax, channel_list, currents, t_model, v_model)
    ax.set_yticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    ax.get_yaxis().set_label_coords(-0.15, 0.5)

    # ionic gates
    ax = pl.Subplot(fig, outer[0, 1])
    fig.add_subplot(ax)

    #plot_gates_on_ax(ax, gates, t_model, v_model)
    plot_product_gates_on_ax(ax, channel_list, gates, t_model, v_model, power_gates)
    ax.get_yaxis().set_label_coords(-0.15, 0.5)

    # activation and inactivation Nat
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0, 2])
    ax = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax)

    v_range = np.arange(-95, 30, 0.1)
    curve_act = boltzmann_fun(v_range, cell.soma(.5).nat.m_vh, -cell.soma(.5).nat.m_vs)
    curve_inact = boltzmann_fun(v_range, cell.soma(.5).nat.h_vh, -cell.soma(.5).nat.h_vs)

    plot_act_inact_on_ax(ax, v_range, curve_act, curve_inact, 'nat')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)

    # blocking ion channels
    axes = [outer[1, 0], outer[1, 1], outer[1, 2]]
    percent_blocks = [10, 50, 100]
    channel_list.remove('pas')

    for percent_block_idx, percent_block in enumerate(percent_blocks):
        ax = pl.Subplot(fig, axes[percent_block_idx])
        fig.add_subplot(ax)

        v_after_block = np.zeros((len(channel_list), len(t_model)))
        for i, channel_name in enumerate(channel_list):
            cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))
            block_channel(cell, channel_name, percent_block)
            v_after_block[i, :], _, _ = simulate_model(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init)

        plot_channel_block_on_ax(ax, channel_list, t_model, v_model, v_after_block, percent_block)
        ax.set_ylim(-90, 60)
        ax.get_yaxis().set_label_coords(-0.15, 0.5)

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'dap_mechanism.png'))
    pl.show()