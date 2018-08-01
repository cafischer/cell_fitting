import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec

from nrn_wrapper import Cell
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model, simulate_model_currents, simulate_model_gates
from cell_fitting.optimization.evaluation.plot_currents import plot_currents_on_ax
from cell_fitting.optimization.evaluation.plot_gates import plot_gates_on_ax
pl.style.use('paper_subplots')


# TODO: colors
# TODO: check simulation_params (e.g. dt)
if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/DAP-Project/thesis/figures'
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
    fig = pl.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(2, 5)

    # simulate for ionic currents and gates
    ramp_amp = 3.5
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell+'.dat'), 'rampIV', ramp_amp)
    v_data += - v_data[0] + v_init
    v_model, t_model, _ = simulate_model(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init)
    currents, channel_list = simulate_model_currents(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init)
    gates, power_gates = simulate_model_gates(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init)

    # ionic currents
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0, 0])
    ax0 = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    plot_currents_on_ax(ax0, channel_list, currents, t_model, v_model)

    # ionic gates
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0, 1])
    ax0 = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    plot_gates_on_ax(ax0,channel_list, gates, t_model, v_model)

    # blocking ion channels
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, 0])
    ax0 = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax0)


    # # blocking Nat and reconstructing AP
    # inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, 1], hspace=0.1)
    # ax0 = pl.Subplot(fig, inner[0])
    # fig.add_subplot(ax0)
    #
    # with open(os.path.join(save_dir_model, model, 'img', 'IV', 'sag', 'sag_dict.json'), 'r') as f:
    #     sag_dict_model = json.load(f)
    #
    # plot_sag_vs_steady_state_on_ax(ax0, color_lines='steelblue', label=False, **sag_dict_model)

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'reproduction_stellate.png'))
    pl.show()