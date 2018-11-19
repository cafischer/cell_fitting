import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.plot_double_ramp import plot_current_threshold_on_ax
from cell_fitting.data.plot_doubleramp import get_inj_doubleramp_params, get_i_inj_double_ramp_full
from cell_fitting.optimization.evaluation.plot_double_ramp.plot_doubleramp_summary import plot_current_threshold_all_cells_on_ax
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    exp_cell_dr = '2015_08_06d'
    color_exp = '#0099cc'
    color_model = 'k'

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'), mechanism_dir)

    fig = pl.figure(figsize=(10.5, 4))
    outer = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.3])

    # double-ramp: mem. pot.
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.1, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    run_idx = 0
    double_ramp_params = get_inj_doubleramp_params(exp_cell_dr, run_idx,
                                                   PP_params_dir='/home/cf/Phd/DAP-Project/cell_data/PP_params2.csv')
    ramp3_amp_exp = double_ramp_params['ramp3_amps'][0]
    ramp3_amp_model = 1.0
    double_ramp_params['ramp3_amps'] = np.arange(1.0, 2.0, 0.05)
    i_inj_mat = get_i_inj_double_ramp_full(**double_ramp_params)

    ramp3_times = double_ramp_params['ramp3_times']
    step_amps = double_ramp_params['step_amps']
    t_exp = np.arange(0, double_ramp_params['tstop'] + double_ramp_params['dt'], double_ramp_params['dt'])

    v_mat_exp = np.load(os.path.join(os.path.join(save_dir_data_plots, 'PP', exp_cell_dr, str(run_idx)), 'v_mat.npy'))

    v_mat_model = np.load(os.path.join(save_dir_model, model, 'img', 'PP', str(int(double_ramp_params['len_step'])),
                                       'v_mat.npy'))

    start_i_inj = np.where(i_inj_mat[0, 0, :, 0] > -0.05)[0][0]
    ramp3_amp_idx_exp = 0
    for ramp3_times_idx in range(len(ramp3_times)):
        # ax0.plot(t_exp, v_mat_exp[ramp3_amp_idx_exp, ramp3_times_idx, :, 1], color_exp,
        #          label='Data' if ramp3_times_idx == 0 else '')
        vrest_data = np.mean(v_mat_exp[ramp3_amp_idx_exp, ramp3_times_idx, :start_i_inj, 1])
        ax0.plot(t_exp, v_mat_exp[ramp3_amp_idx_exp, ramp3_times_idx, :, 1] - vrest_data, color_exp,
                 label='Data' if ramp3_times_idx == 0 else '')
    ax0.set_xlim(470, 530)
    ax0.set_xticks([])

    ramp3_amp_idx_model = 8
    for ramp3_times_idx in range(len(ramp3_times)):
        # ax0.plot(t_exp, v_mat_model[ramp3_amp_idx_model, ramp3_times_idx, :, 1], color_model,
        #          label='Model' if ramp3_times_idx == 0 else '')
        vrest_model = np.mean(v_mat_model[ramp3_amp_idx_model, ramp3_times_idx, :start_i_inj, 1])
        ax0.plot(t_exp, v_mat_model[ramp3_amp_idx_model, ramp3_times_idx, :, 1] - vrest_model, color_model,
                 label='Model' if ramp3_times_idx == 0 else '')
    ax0.set_xlim(470, 530)
    ax0.set_xticks([])

    for ramp3_times_idx in range(len(ramp3_times)):
        ax1.plot(t_exp, i_inj_mat[np.where(np.isclose(ramp3_amp_exp + ramp3_amp_idx_exp * 0.05, double_ramp_params['ramp3_amps']))[0][0],
                        ramp3_times_idx, :, 1], color_exp)
        ax1.plot(t_exp, i_inj_mat[np.where(np.isclose(ramp3_amp_model + ramp3_amp_idx_model * 0.05, double_ramp_params['ramp3_amps']))[0][0],
                        ramp3_times_idx, :, 1], color_model)
    ax1.set_xlim(470, 530)

    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.15, 0.5)
    ax1.get_yaxis().set_label_coords(-0.15, 0.5)
    ax0.legend()
    ax0.text(-0.23, 1.0, 'A', transform=ax0.transAxes, size=18, weight='bold')

    # double-ramp: current threshold
    ax0 = pl.Subplot(fig, outer[0, 1])
    fig.add_subplot(ax0)

    with open(os.path.join(save_dir_model, model, 'img', 'PP', str(int(double_ramp_params['len_step'])), 
                           'current_threshold_dict.json'), 'r') as f:
        current_threshold_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'PP', exp_cell_dr, 'current_threshold_dict.json'), 'r') as f:
        current_threshold_dict_data = json.load(f)

    # # transform to percentages
    # current_threshold_dict_model['current_thresholds'] = np.array(current_threshold_dict_model['current_thresholds']) \
    #                                                      / current_threshold_dict_model['current_threshold_rampIV'] * 100
    # current_threshold_dict_model['current_threshold_rampIV'] = 100
    #
    # current_threshold_dict_data['current_thresholds'] = np.array(current_threshold_dict_data['current_thresholds']) \
    #                                                      / current_threshold_dict_data['current_threshold_rampIV'] * 100
    # current_threshold_dict_data['current_threshold_rampIV'] = 100

    plot_current_threshold_on_ax(ax0, colors_dict={-0.1: color_model, 0.0: color_model, 0.1: color_model},
                                 label=False, plot_range=False, with_right_spine=False, shift_to_rest=True,
                                 **current_threshold_dict_model)
    plot_current_threshold_on_ax(ax0, colors_dict={-0.1: color_exp, 0.0: color_exp, 0.1: color_exp}, label=False,
                                 with_right_spine=False, shift_to_rest=True, **current_threshold_dict_data)
    ax0.get_yaxis().set_label_coords(-0.15, 0.5)

    # legend
    custom_lines = [
                    Line2D([0], [0], marker='^', color='None', markerfacecolor='0.5', markeredgecolor='0.5', lw=1.0),
                    Line2D([0], [0], marker='o', color='None', markerfacecolor='0.5', markeredgecolor='0.5', lw=1.0),
                    Line2D([0], [0], marker='v', color='None', markerfacecolor='0.5', markeredgecolor='0.5', lw=1.0)]
    ax0.legend(custom_lines, ['Amp.: 0.1', 'Amp.: 0', 'Amp.: -0.1'], loc='upper right')
    ax0.text(-0.24, 1.0, 'B', transform=ax0.transAxes, size=18, weight='bold')

    # comparison current threshold DAP - rest all cells
    ax = pl.Subplot(fig, outer[0, 2])
    fig.add_subplot(ax)

    with open(os.path.join(save_dir_model, model, 'img', 'PP', '125', 'current_threshold_dict.json'), 'r') as f:
        current_threshold_dict_model_short_step = json.load(f)

    cell_ids = ['2014_07_10b', '2014_07_03a', '2014_07_08d', '2014_07_09c', '2014_07_09e', '2014_07_09f', '2014_07_10d']
    current_thresholds_DAP = np.zeros((len(cell_ids), 3))
    current_thresholds_rest = np.zeros(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        current_thresholds_DAP[cell_idx, :] = np.loadtxt(os.path.join(save_dir_data_plots, 'PP', cell_id,
                                                                       'current_threshold_DAP.txt'))
        current_thresholds_rest[cell_idx] = float(np.loadtxt(os.path.join(save_dir_data_plots, 'PP', cell_id,
                                                                       'current_threshold_rest.txt')))
    mean_percentage_difference_exp = plot_current_threshold_all_cells_on_ax(ax, current_thresholds_DAP,
                                                                            current_thresholds_rest,
                                                                            current_threshold_dict_data['step_amps'],
                                                                            color=color_exp,
                                                                            plot_sig=False)

    current_threshold_DAP_model = np.nanmin(current_threshold_dict_model_short_step['current_thresholds'], axis=1)
    current_threshold_rest_model = current_threshold_dict_model_short_step['current_threshold_rampIV']
    percentage_difference_model = 100 - (current_threshold_DAP_model / current_threshold_rest_model * 100)
    ax.plot(-0.4, percentage_difference_model[0], 'o', color=color_model, alpha=0.5)
    ax.plot(0.6, percentage_difference_model[1], 'o', color=color_model, alpha=0.5)
    ax.plot(1.6, percentage_difference_model[2], 'o', color=color_model, alpha=0.5)
    ax.get_yaxis().set_label_coords(-0.3, 0.5)
    ax.text(-0.60, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')

    print 'percentage difference model: ', percentage_difference_model
    print 'mean percentage difference real cells: ', mean_percentage_difference_exp

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'reproduction_double_ramp.png'))
    pl.show()