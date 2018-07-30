import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec

from nrn_wrapper import Cell
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import plot_sag_vs_steady_state_on_ax
from cell_fitting.optimization.evaluation.plot_double_ramp import plot_current_threshold_on_ax
from cell_fitting.optimization.evaluation.plot_IV import plot_fi_curve_on_ax
from cell_fitting.optimization.evaluation.plot_zap import plot_impedance_on_ax
pl.style.use('paper')


# TODO: colors
# TODO: style for figures with many subpanels
if __name__ == '__main__':
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    v_init = -75

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'), mechanism_dir)

    fig = pl.figure(figsize=(12, 8))
    outer = gridspec.GridSpec(2, 5)

    # DAP
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.1, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    ramp_amp = 3.5
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell+'.dat'), 'rampIV', ramp_amp)
    v_data += - v_data[0] + v_init
    v_model, t_model, _ = simulate_model(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init)

    ax0.plot(t_data, v_data, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, 'steelblue', label='Model')
    ax1.plot(t_data, i_inj, 'k')

    ax0.legend(fontsize=12)
    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)', fontsize=12)
    ax1.set_ylabel('Inj. current (nA)', fontsize=12)
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])
    ax0.xaxis.set_tick_params(labelsize=10)
    ax0.yaxis.set_tick_params(labelsize=10)
    ax1.xaxis.set_tick_params(labelsize=10)
    ax1.yaxis.set_tick_params(labelsize=10)

    # double-ramp
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 0])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    with open(os.path.join(save_dir_model, model, 'img', 'PP', '125', 'current_threshold_dict.json'), 'r') as f:
        current_threshold_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'PP', '2014_07_10b', 'current_threshold_dict.json'), 'r') as f:  # TODO: here using different cell!!!
        current_threshold_dict_data = json.load(f)

    plot_current_threshold_on_ax(ax0, **current_threshold_dict_model)
    plot_current_threshold_on_ax(ax1, **current_threshold_dict_data)
    # TODO: same y limits both sides

    ax0.set_xlabel('')
    ax0.set_xticks([])

    # sag
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], hspace=0.1, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    step_amp = -0.1
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'IV', step_amp)
    v_data += - v_data[0] + v_init
    v_model, t_model, _ = simulate_model(cell, 'IV', step_amp, t_data[-1], v_init=v_init)

    ax0.plot(t_data, v_data, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, 'steelblue', label='Model')
    ax1.plot(t_data, i_inj, 'k')

    ax0.legend(fontsize=12)
    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)', fontsize=12)
    ax1.set_ylabel('Inj. current (nA)', fontsize=12)
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])
    ax0.xaxis.set_tick_params(labelsize=10)
    ax0.yaxis.set_tick_params(labelsize=10)
    ax1.xaxis.set_tick_params(labelsize=10)
    ax1.yaxis.set_tick_params(labelsize=10)

    # sag vs. steady-state
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 1], hspace=0.1)
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    with open(os.path.join(save_dir_model, model, 'img', 'IV', 'sag', 'sag_dict.json'), 'r') as f:
        sag_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'IV', 'sag', exp_cell, 'sag_dict.json'), 'r') as f:
        sag_dict_data = json.load(f)

    plot_sag_vs_steady_state_on_ax(ax0, **sag_dict_model)
    plot_sag_vs_steady_state_on_ax(ax1, **sag_dict_data)

    ax0.set_xlabel('')
    ax0.set_xticks([])

    # pos. step
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 2], hspace=0.1, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    step_amp = 0.4
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'IV', step_amp)
    v_data += - v_data[0] + v_init
    v_model, t_model, _ = simulate_model(cell, 'IV', step_amp, t_data[-1], v_init=v_init)

    ax0.plot(t_data, v_data, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, 'steelblue', label='Model')
    ax1.plot(t_data, i_inj, 'k')

    ax0.legend(fontsize=12)
    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)', fontsize=12)
    ax1.set_ylabel('Inj. current (nA)', fontsize=12)
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])
    ax0.xaxis.set_tick_params(labelsize=10)
    ax0.yaxis.set_tick_params(labelsize=10)
    ax1.xaxis.set_tick_params(labelsize=10)
    ax1.yaxis.set_tick_params(labelsize=10)

    # f-I curve
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 2], hspace=0.1)
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    with open(os.path.join(save_dir_model, model, 'img', 'IV', 'fi_curve', 'fi_dict.json'), 'r') as f:
        fi_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'IV', 'fi_curve', exp_cell, 'fi_dict.json'), 'r') as f:
        fi_dict_data = json.load(f)

    plot_fi_curve_on_ax(ax0, **fi_dict_model)
    plot_fi_curve_on_ax(ax1, **fi_dict_data)

    ax0.set_xlabel('')
    ax0.set_xticks([])

    # zap
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 3], hspace=0.1, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    step_amp = 0.1
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'Zap20', step_amp)
    v_data += - v_data[0] + v_init
    v_model, t_model, _ = simulate_model(cell, 'Zap20', step_amp, t_data[-1], v_init=v_init, dt=0.025)

    ax0.plot(t_data, v_data, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, 'steelblue', label='Model')
    ax1.plot(t_data, i_inj, 'k')

    ax0.legend(fontsize=12)
    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)', fontsize=12)
    ax1.set_ylabel('Inj. current (nA)', fontsize=12)
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])
    ax0.xaxis.set_tick_params(labelsize=10)
    ax0.yaxis.set_tick_params(labelsize=10)
    ax1.xaxis.set_tick_params(labelsize=10)
    ax1.yaxis.set_tick_params(labelsize=10)

    # impedance
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 3], hspace=0.1)
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    with open(os.path.join(save_dir_model, model, 'img', 'zap', 'impedance_dict.json'), 'r') as f:
        impedance_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'Zap20', exp_cell, 'impedance_dict.json'), 'r') as f:
        impedance_dict_data = json.load(f)

    plot_impedance_on_ax(ax0, **impedance_dict_model)
    plot_impedance_on_ax(ax1, **impedance_dict_data)
    # TODO: add legend with res. freq. and q-value

    ax0.set_xlabel('')
    ax0.set_xticks([])

    # sine

    # phase hist.


    pl.tight_layout()
    #pl.savefig(save_dir_img)
    pl.show()