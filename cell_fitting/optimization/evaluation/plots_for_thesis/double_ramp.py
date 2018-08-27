import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.gridspec as gridspec
import numpy as np
import os
import json
from cell_fitting.data.plot_doubleramp import get_inj_doubleramp_params, get_i_inj_double_ramp_full
from cell_fitting.optimization.evaluation.plot_double_ramp import plot_current_threshold_on_ax
from cell_fitting.optimization.evaluation.plot_double_ramp.plot_doubleramp_summary import plot_current_threshold_all_cells_on_ax
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

    run_idx = 0
    double_ramp_params = get_inj_doubleramp_params(cell_id, run_idx,
                                                   PP_params_dir='/home/cf/Phd/DAP-Project/cell_data/PP_params2.csv')
    ramp3_amp_exp = double_ramp_params['ramp3_amps'][0]
    ramp3_amp_model = 1.0
    double_ramp_params['ramp3_amps'] = np.arange(1.0, 2.0, 0.05)
    i_inj_mat = get_i_inj_double_ramp_full(**double_ramp_params)

    ramp3_times = double_ramp_params['ramp3_times']
    step_amps = double_ramp_params['step_amps']
    t_exp = np.arange(0, double_ramp_params['tstop'] + double_ramp_params['dt'], double_ramp_params['dt'])

    v_mat = np.load(os.path.join(os.path.join(save_dir_data_plots, 'PP', cell_id, str(run_idx)), 'v_mat.npy'))

    # plot
    fig = pl.figure(figsize=(10.0, 6.0))
    outer = gridspec.GridSpec(2, 3, width_ratios=[1, 0.4, 0.75])
    
    # voltage
    ax = pl.Subplot(fig, outer[0, 0])
    fig.add_subplot(ax)
    for ramp3_time_idx in range(0, len(ramp3_times)):
        ax.plot(t_exp, v_mat[0, ramp3_time_idx, :, 2], '--r', linewidth=0.8)
    for ramp3_time_idx in range(0, len(ramp3_times)):
        ax.plot(t_exp, v_mat[0, ramp3_time_idx, :, 0], '--b', linewidth=0.8)
    for ramp3_time_idx in range(0, len(ramp3_times)):
        ax.plot(t_exp, v_mat[0, ramp3_time_idx, :, 1], '--k', linewidth=0.8)
    ax.plot(t_exp, v_mat[0, 0, :, 2], c='r', linewidth=1.0)
    ax.plot(t_exp, v_mat[0, 0, :, 0], c='b', linewidth=1.0)
    ax.plot(t_exp, v_mat[0, 0, :, 1], c='k', linewidth=1.0)

    axins = inset_axes(ax, width='65%', height='100%',
                       bbox_to_anchor=(1.0, 0.2, 1.0, 0.75),
                       bbox_transform=ax.transAxes, loc=3)
    for ramp3_time_idx in range(len(ramp3_times)):
        axins.plot(t_exp, v_mat[0, ramp3_time_idx, :, 2], '--r', linewidth=0.8)
    for ramp3_time_idx in range(len(ramp3_times)):
        axins.plot(t_exp, v_mat[0, ramp3_time_idx, :, 0], '--b', linewidth=0.8)
    for ramp3_time_idx in range(len(ramp3_times)):
        axins.plot(t_exp, v_mat[0, ramp3_time_idx, :, 1], '--k', linewidth=0.8)
    axins.plot(t_exp, v_mat[0, 0, :, 2], c='r', linewidth=1.0)
    axins.plot(t_exp, v_mat[0, 0, :, 0], c='b', linewidth=1.0)
    axins.plot(t_exp, v_mat[0, 0, :, 1], c='k', linewidth=1.0)
    axins.set_xlim(470, 530)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mem. pot. (nA)')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.23, 1.0, 'A', transform=ax.transAxes, size=18, weight='bold')

    # current
    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)
    for ramp3_time_idx in range(1, len(ramp3_times)):
        ax.plot(t_exp, i_inj_mat[0, ramp3_time_idx, :, 1], '--k', linewidth=0.8)
    ax.plot(t_exp, i_inj_mat[0, 0, :, 2], c='r', linewidth=1.0)
    ax.plot(t_exp, i_inj_mat[0, 0, :, 0], c='b', linewidth=1.0)
    ax.plot(t_exp, i_inj_mat[0, 0, :, 1], c='k', linewidth=1.0)

    axins = inset_axes(ax, width='65%', height='100%',
                       bbox_to_anchor=(1.0, 0.2, 1.0, 0.75),
                       bbox_transform=ax.transAxes, loc=3)
    for ramp3_time_idx in range(1, len(ramp3_times)):
        axins.plot(t_exp, i_inj_mat[0, ramp3_time_idx, :, 1], '--k', linewidth=0.8)
    axins.plot(t_exp, i_inj_mat[0, 0, :, 2], c='r', linewidth=1.0)
    axins.plot(t_exp, i_inj_mat[0, 0, :, 0], c='b', linewidth=1.0)
    axins.plot(t_exp, i_inj_mat[0, 0, :, 1], c='k', linewidth=1.0)
    axins.set_xlim(470, 530)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (nA)')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.23, 1.0, 'B', transform=ax.transAxes, size=18, weight='bold')

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

    ax.text(-0.24, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')

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

    ax.text(-2.5, 1.0, 'D', transform=ax.transAxes, size=18, weight='bold')

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'double_ramp.png'))
    pl.show()