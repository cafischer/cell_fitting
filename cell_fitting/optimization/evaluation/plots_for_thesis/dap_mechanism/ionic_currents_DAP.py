import numpy as np
import os
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb
from cell_fitting.util import change_color_brightness
from nrn_wrapper import Cell
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model, simulate_model_currents, simulate_model_gates
from cell_fitting.optimization.evaluation.plot_currents import plot_currents_on_ax
from cell_fitting.optimization.evaluation.plot_gates import plot_product_gates_on_ax, plot_gates_on_ax
from cell_fitting.optimization.evaluation.plot_blocking.block_channel import block_channel_at_timepoint
from cell_fitting.test_channels.channel_characteristics import boltzmann_fun, time_constant_curve
from cell_fitting.util import get_channel_dict_for_plotting, get_channel_color_for_plotting
pl.style.use('paper_subplots')



# TODO: check simulation params
if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    v_init = -75
    onset = 200
    dt = 0.01
    ramp_amp = 3.5

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'), mechanism_dir)

    # plot 1
    fig = pl.figure(figsize=(8, 6))
    outer = gridspec.GridSpec(2, 2)

    # simulate ionic currents and gates (without block)
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell+'.dat'), 'rampIV', ramp_amp)
    v_data += - v_data[0] + v_init
    v_model, t_model, _ = simulate_model(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init, onset=200, dt=dt)
    currents, channel_list = simulate_model_currents(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init, onset=200, dt=dt)
    gates, power_gates = simulate_model_gates(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init, onset=200, dt=dt)

    # ionic currents
    ax = pl.Subplot(fig, outer[0, 0])
    fig.add_subplot(ax)

    plot_currents_on_ax(ax, channel_list, currents, t_model, v_model)
    ax.set_yticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.25, 1.0, 'A', transform=ax.transAxes, size=18, weight='bold')

    # ionic gates
    ax = pl.Subplot(fig, outer[0, 1])
    fig.add_subplot(ax)

    plot_gates_on_ax(ax, gates, t_model, v_model)
    #plot_product_gates_on_ax(ax, channel_list, gates, t_model, v_model, power_gates)
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.25, 1.0, 'B', transform=ax.transAxes, size=18, weight='bold')

    # simulate for ionic currents and gates (with block)
    t_block_idx = 1398
    block_channel_at_timepoint(cell, 'hcn_slow', 100, t_model[t_block_idx] + onset)
    block_channel_at_timepoint(cell, 'nat', 100, t_model[t_block_idx] + onset)
    v_model, t_model, _ = simulate_model(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init, onset=200, dt=dt)

    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))
    block_channel_at_timepoint(cell, 'hcn_slow', 100, t_model[t_block_idx] + onset)
    block_channel_at_timepoint(cell, 'nat', 100, t_model[t_block_idx] + onset)
    currents, channel_list = simulate_model_currents(cell, 'rampIV', ramp_amp, t_data[-1],
                                                     v_init=v_init, onset=200, dt=dt)

    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))
    block_channel_at_timepoint(cell, 'hcn_slow', 100, t_model[t_block_idx] + onset)
    block_channel_at_timepoint(cell, 'nat', 100, t_model[t_block_idx] + onset)
    gates, power_gates = simulate_model_gates(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init, onset=200, dt=dt)

    # ionic currents
    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)

    nat_idx = np.where(np.array(channel_list) == 'nat')[0][0]
    hcn_idx = np.where(np.array(channel_list) == 'hcn_slow')[0][0]
    currents[nat_idx][t_block_idx:] = np.nan
    currents[hcn_idx][t_block_idx:] = np.nan
    gates['nat_m'][t_block_idx:] = np.nan
    gates['nat_h'][t_block_idx:] = np.nan
    gates['hcn_slow_n'][t_block_idx:] = np.nan
    # channel_list.remove('nat')
    # channel_list.remove('hcn_slow')
    # currents = np.delete(currents, (nat_idx, hcn_idx), axis=0)
    # gates.pop('nat_m')
    # gates.pop('nat_h')
    # gates.pop('hcn_slow_n')

    plot_currents_on_ax(ax, channel_list, currents, t_model, v_model)
    # ax.plot(t_model - 7, -1 * (currents[np.array(channel_list) == 'nap'][0]
    #                            + currents[np.array(channel_list) == 'kdr'][0]
    #                            + currents[np.array(channel_list) == 'pas'][0]
    #                            ), color='magenta')
    # ax.plot(t_model - 7, -1 * (currents[np.array(channel_list) == 'nap'][0]
    #                            + currents[np.array(channel_list) == 'kdr'][0]
    #                            ), color='orange')
    ax.set_yticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.25, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')

    # ionic gates
    ax = pl.Subplot(fig, outer[1, 1])
    fig.add_subplot(ax)

    plot_gates_on_ax(ax, gates, t_model, v_model)
    #plot_product_gates_on_ax(ax, channel_list, gates, t_model, v_model, power_gates)
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.25, 1.0, 'D', transform=ax.transAxes, size=18, weight='bold')


    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'ionic_currents_DAP.png'))
    pl.show()