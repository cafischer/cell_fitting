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
from cell_fitting.optimization.evaluation.plot_sine_stimulus import simulate_sine_stimulus
from grid_cell_stimuli.spike_phase import plot_phase_hist_on_axes
pl.style.use('paper_subplots')


# TODO: colors
# TODO: check simulation_params (e.g. dt)
# TODO: check all exp. data are v_shifted
# TODO: check same protocol used for double ramp
if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/DAP-Project/thesis/figures'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'  # TODO: choose model 2
    exp_cell = '2015_08_26b'
    v_init = -75
    color_model = '0.5'

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'), mechanism_dir)

    fig = pl.figure(figsize=(11, 6.0))
    outer = gridspec.GridSpec(2, 5)

    # DAP
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.2, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    ramp_amp = 3.5
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell+'.dat'), 'rampIV', ramp_amp)
    #v_data += - v_data[0] + v_init
    v_model, t_model, _ = simulate_model(cell, 'rampIV', ramp_amp, t_data[-1], v_init=v_init)

    ax0.plot(t_data, v_data, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, color_model, label='Model')
    ax1.plot(t_data, i_inj, 'k')

    ax0.legend()
    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])

    # double-ramp
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, 0], hspace=0.2)
    ax0 = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    with open(os.path.join(save_dir_model, model, 'img', 'PP', '125', 'current_threshold_dict.json'), 'r') as f:
        current_threshold_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'PP', '2015_08_06d', 'current_threshold_dict.json'), 'r') as f:  # using different cell here! 2014_07_10b
        current_threshold_dict_data = json.load(f)

    plot_current_threshold_on_ax(ax0, colors_dict={-0.1: color_model, 0.0: color_model, 0.1: color_model},
                                 label=False, plot_range=False, **current_threshold_dict_model)
    plot_current_threshold_on_ax(ax0, colors_dict={-0.1: 'k', 0.0: 'k', 0.1: 'k'}, label=True,
                                 **current_threshold_dict_data)
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)

    # sag
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], hspace=0.2, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    step_amp = -0.1
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'IV', step_amp)
    # v_data += - v_data[0] + v_init
    v_model, t_model, _ = simulate_model(cell, 'IV', step_amp, t_data[-1], v_init=v_init)

    ax0.plot(t_data, v_data, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, color_model, label='Model')
    ax1.plot(t_data, i_inj, 'k')

    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])

    # sag vs. steady-state
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, 1], hspace=0.2)
    ax0 = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    with open(os.path.join(save_dir_model, model, 'img', 'IV', 'sag', 'sag_dict.json'), 'r') as f:
        sag_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'IV', 'sag', exp_cell, 'sag_dict.json'), 'r') as f:
        sag_dict_data = json.load(f)

    plot_sag_vs_steady_state_on_ax(ax0, color_lines=color_model, label=False, **sag_dict_model)
    plot_sag_vs_steady_state_on_ax(ax0, color_lines='k', label=True, **sag_dict_data)

    ax0.get_yaxis().set_label_coords(-0.25, 0.5)

    # pos. step
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 2], hspace=0.2, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    step_amp = 0.4
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'IV', step_amp)
    # v_data += - v_data[0] + v_init
    v_model, t_model, _ = simulate_model(cell, 'IV', step_amp, t_data[-1], v_init=v_init)

    ax0.plot(t_data, v_data, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, color_model, label='Model')
    ax1.plot(t_data, i_inj, 'k')

    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])

    # f-I curve
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, 2], hspace=0.2)
    ax0 = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    with open(os.path.join(save_dir_model, model, 'img', 'IV', 'fi_curve', 'fi_dict.json'), 'r') as f:
        fi_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'IV', 'fi_curve', 'rat', exp_cell, 'fi_dict.json'), 'r') as f:
        fi_dict_data = json.load(f)

    plot_fi_curve_on_ax(ax0, color_line=color_model, **fi_dict_model)
    plot_fi_curve_on_ax(ax0, color_line='k', **fi_dict_data)

    ax0.get_yaxis().set_label_coords(-0.25, 0.5)

    # zap
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 3], hspace=0.2, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    step_amp = 0.1
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'Zap20', step_amp)
    # v_data += - v_data[0] + v_init
    v_model, t_model, _ = simulate_model(cell, 'Zap20', step_amp, t_data[-1], v_init=v_init, dt=0.025)

    ax0.plot(t_data, v_data, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, color_model, label='Model')
    ax1.plot(t_data, i_inj, 'k')

    ax0.set_xticks([])
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.get_yaxis().set_label_coords(-0.25, 0.5)

    # impedance
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, 3], hspace=0.2)
    ax0 = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    with open(os.path.join(save_dir_model, model, 'img', 'zap', 'impedance_dict.json'), 'r') as f:
        impedance_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'Zap20', exp_cell, 'impedance_dict.json'), 'r') as f:
        impedance_dict_data = json.load(f)

    plot_impedance_on_ax(ax0, color_line=color_model, **impedance_dict_model)
    plot_impedance_on_ax(ax0, color_line='k', **impedance_dict_data)
    # TODO: add legend with res. freq. and q-value

    ax0.get_yaxis().set_label_coords(-0.25, 0.5)

    # sine
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 4], hspace=0.2, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    amp1 = 0.4
    amp2 = 0.2
    freq1 = 0.1
    freq2 = 5
    s_ = os.path.join(save_dir_data_plots, 'sine_stimulus/traces/rat', '2015_08_20d',  # using different cell here!
                      str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2))
    v_data = np.load(os.path.join(s_, 'v.npy'))
    t_data = np.load(os.path.join(s_, 't.npy'))
    # v_data += - v_data[0] + v_init
    v_model, t_model, i_inj = simulate_sine_stimulus(cell, amp1, amp2, 1./freq1*1000/2., freq2, 500, 500,
                                                     t_data[1]-t_data[0], v_init=-75, celsius=35, onset=200)

    ax0.plot(t_data, v_data, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, color_model, label='Model')
    ax1.plot(t_model, i_inj, 'k')

    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.set_yticks([np.round(np.min(i_inj), 1), np.round(np.max(i_inj), 1)])

    # phase hist.
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, 4], hspace=0.2)
    ax0 = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    with open(os.path.join(save_dir_model, model, 'img', 'sine_stimulus/traces',
                           str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2), 'phase_hist',
                           'sine_dict.json'), 'r') as f:
        sine_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'sine_stimulus/traces/rat', '2015_08_20d',  # using different cell here!
                      str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2),
                           'spike_phase', 'sine_dict.json'), 'r') as f:
        sine_dict_data = json.load(f)

    plot_phase_hist_on_axes(ax0, 0, [sine_dict_model['phases']], plot_mean=True, color_hist=color_model,
                            alpha=0.6, color_lines=color_model)
    plot_phase_hist_on_axes(ax0, 0, [sine_dict_data['phases']], plot_mean=True, color_hist='k',
                            alpha=0.3, color_lines='k')

    ax0.set_ylabel('Count')
    ax0.set_xlabel('Phase (deg.)')
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'reproduction_stellate.png'))
    pl.show()