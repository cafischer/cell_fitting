import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.gridspec as gridspec
import numpy as np
import os
import json
import pandas as pd
from cell_fitting.optimization.evaluation.plot_double_ramp.plot_doubleramp import get_ramp3_times
from cell_fitting.read_heka.i_inj_functions import get_i_inj_double_ramp
from cell_fitting.optimization.evaluation.plot_double_ramp import plot_current_threshold_on_ax
from cell_fitting.optimization.evaluation.plot_double_ramp.plot_doubleramp_summary import plot_current_threshold_all_cells_on_ax
from cell_characteristics import to_idx
pl.style.use('paper_subplots')


# TODO: ramp3amp same for current and v
# TODO: include also 2015 cells !
# TODO: could do plot in lower right also for hyper/depolarization
if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_introduction'
    cell_id = '2015_08_06d'
    run_idx = 0
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    save_dir_cell = os.path.join(save_dir_data_plots, 'PP', cell_id, str(run_idx))
    PP_params_dir = '/home/cf/Phd/DAP-Project/cell_data/PP_params2.csv'
    PP_params = pd.read_csv(PP_params_dir, header=0)
    PP_params['cell_id'].fillna(method='ffill', inplace=True)
    params = PP_params[PP_params['cell_id'] == cell_id].iloc[run_idx]
    dt = 0.01
    tstop = 691.99
    t_exp = np.arange(0, tstop+dt, dt)
    len_step = params['step_len']
    step_amps = [0, 0.1, -0.1]
    ramp_amp = params['ramp2_amp']
    ramp3_amp = params['ramp3_amp']
    baseline_amp = -0.05
    ramp3_times = get_ramp3_times(params['delta_first'], params['delta_ramp'], params['len_ramp3_times'])
    len_step2ramp = params['len_step2ramp']
    len_ramp = 2

    start_step = to_idx(222, dt)
    end_step = start_step + to_idx(len_step, dt)
    start_ramp2 = params['start_ramp2_idx']

    i_inj_mat = np.zeros([len(ramp3_times), len(t_exp), len(step_amps)])
    v_mat = np.zeros([len(ramp3_times), len(t_exp), len(step_amps)])
    for k, step_amp in enumerate(step_amps):
        step_str = 'step_%.1f(nA)' % step_amp
        for i, ramp3_time in enumerate(ramp3_times):
            v_mat[i, :, k] = np.load(os.path.join(save_dir_cell, step_str, 'v_mat.npy'))[0, i, :]

            i_inj_mat[i, :, k] = get_i_inj_double_ramp(ramp_amp, ramp3_amp, ramp3_time, step_amp, len_step,
                                                       baseline_amp, len_ramp, len_step2ramp=len_step2ramp,
                                                       tstop=tstop, dt=dt)


    # plot
    fig = pl.figure(figsize=(10.0, 6.0))
    outer = gridspec.GridSpec(2, 3, width_ratios=[1, 0.4, 0.75])
    
    # voltage
    ax = pl.Subplot(fig, outer[0, 0])
    fig.add_subplot(ax)
    for i in range(0, len(ramp3_times)):
        ax.plot(t_exp, v_mat[i, :, 0], '--k', linewidth=0.8)
    for i in range(0, len(ramp3_times)):
        ax.plot(t_exp, v_mat[i, :, 1], '--r', linewidth=0.8)
    for i in range(0, len(ramp3_times)):
        ax.plot(t_exp, v_mat[i, :, 2], '--b', linewidth=0.8)
    ax.plot(t_exp, v_mat[0, :, 1], c='r', linewidth=1.0)
    ax.plot(t_exp, v_mat[0, :, 2], c='b', linewidth=1.0)
    ax.plot(t_exp, v_mat[0, :, 0], c='k', linewidth=1.0)

    axins = inset_axes(ax, width='65%', height='100%',
                       bbox_to_anchor=(1.0, 0.2, 1.0, 0.75),
                       bbox_transform=ax.transAxes, loc=3)
    for i in range(0, len(ramp3_times)):
        axins.plot(t_exp, v_mat[i, :, 0], '--k', linewidth=0.8)
    for i in range(0, len(ramp3_times)):
        axins.plot(t_exp, v_mat[i, :, 1], '--r', linewidth=0.8)
    for i in range(0, len(ramp3_times)):
        axins.plot(t_exp, v_mat[i, :, 2], '--b', linewidth=0.8)
    axins.plot(t_exp, v_mat[0, :, 1], c='r', linewidth=1.0)
    axins.plot(t_exp, v_mat[0, :, 2], c='b', linewidth=1.0)
    axins.plot(t_exp, v_mat[0, :, 0], c='k', linewidth=1.0)
    axins.set_xlim(470, 530)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mem. pot. (nA)')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)

    # current
    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)
    for i in range(1, len(ramp3_times)):
        ax.plot(t_exp, i_inj_mat[i, :, 0], '--k', linewidth=0.8)
    ax.plot(t_exp, i_inj_mat[0, :, 1], c='r', linewidth=1.0)
    ax.plot(t_exp, i_inj_mat[0, :, 2], c='b', linewidth=1.0)
    ax.plot(t_exp, i_inj_mat[0, :, 0], c='k', linewidth=1.0)

    axins = inset_axes(ax, width='65%', height='100%',
                       bbox_to_anchor=(1.0, 0.2, 1.0, 0.75),
                       bbox_transform=ax.transAxes, loc=3)
    for i in range(1, len(ramp3_times)):
        axins.plot(t_exp, i_inj_mat[i, :, 0], '--k', linewidth=0.8)
    axins.plot(t_exp, i_inj_mat[0, :, 1], c='r', linewidth=1.0)
    axins.plot(t_exp, i_inj_mat[0, :, 2], c='b', linewidth=1.0)
    axins.plot(t_exp, i_inj_mat[0, :, 0], c='k', linewidth=1.0)
    axins.set_xlim(470, 530)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (nA)')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)

    # invisible 2nd column
    ax = pl.Subplot(fig, outer[0, 1])
    fig.add_subplot(ax)
    ax.axis('off')
    ax = pl.Subplot(fig, outer[1, 1])
    fig.add_subplot(ax)
    ax.axis('off')

    # current threshold
    ax = pl.Subplot(fig, outer[0, 2])
    fig.add_subplot(ax)
    with open(os.path.join(save_dir_data_plots, 'PP', '2015_08_06d', 'current_threshold_dict.json'), 'r') as f:
        current_threshold_dict_data = json.load(f)

    plot_current_threshold_on_ax(ax, colors_dict = {-0.1: 'b', 0.0: 'k', 0.1: 'r'},
                                 label=True, **current_threshold_dict_data)

    # comparison current threshold DAP - rest all cells
    ax = pl.Subplot(fig, outer[1, 2])
    fig.add_subplot(ax)
    cell_ids = ['2014_07_10b', '2014_07_03a', '2014_07_08d', '2014_07_09c', '2014_07_09e', '2014_07_09f', '2014_07_10d']
    current_thresholds_DAP = np.zeros(len(cell_ids))
    current_thresholds_rest = np.zeros(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        current_thresholds_DAP[cell_idx] = float(np.loadtxt(os.path.join(save_dir_data_plots, 'PP', cell_id,
                                                                       'current_threshold_DAP.txt')))
        current_thresholds_rest[cell_idx] = float(np.loadtxt(os.path.join(save_dir_data_plots, 'PP', cell_id,
                                                                       'current_threshold_rest.txt')))
    plot_current_threshold_all_cells_on_ax(ax, current_thresholds_DAP, current_thresholds_rest)

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'double_ramp.png'))
    pl.show()