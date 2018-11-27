import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
#from mpl_toolkits.mplot3d import Axes3D
from nrn_wrapper import Cell
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model
from cell_fitting.optimization.evaluation.plot_IV import plot_fi_curve_on_ax, simulate_and_compute_fI_curve, \
    fit_fI_curve
from cell_fitting.optimization.evaluation.plot_IV.latency_vs_ISI12_distribution import get_latency_and_ISI12
from cell_fitting.optimization.simulate import get_standard_simulation_params
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    color_exp = '#0099cc'
    color_model = 'k'
    step_amp = 0.4
    standard_sim_params = get_standard_simulation_params()

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'), mechanism_dir)

    fig = pl.figure(figsize=(8, 9))
    outer = gridspec.GridSpec(3, 2)

    # pos. step
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.1, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'IV', step_amp)
    v_model, t_model, _ = simulate_model(cell, 'IV', step_amp, t_data[-1], **standard_sim_params)

    start_i_inj = np.where(i_inj)[0][0]
    vrest_data = np.mean(v_data[:start_i_inj])
    vrest_model = np.mean(v_model[:start_i_inj])

    ax0.plot(t_data, v_data - vrest_data, color_exp, label='Data')
    ax0.plot(t_model, v_model - vrest_model, color_model, label='Model')
    ax1.plot(t_data, i_inj, 'k')

    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.15, 0.5)
    ax1.get_yaxis().set_label_coords(-0.15, 0.5)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])
    ax0.legend()
    ax0.text(-0.25, 1.0, 'A', transform=ax0.transAxes, size=18, weight='bold')

    # latency vs ISI1/2
    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)
    latency_data = np.load(os.path.join(save_dir_data_plots, 'IV/latency_vs_ISI12/rat', 'latency.npy'))
    ISI12_data = np.load(os.path.join(save_dir_data_plots, 'IV/latency_vs_ISI12/rat', 'ISI12.npy'))
    ax.plot(latency_data[latency_data>=0], ISI12_data[latency_data>=0], 'o', color=color_exp,
            alpha=0.5, label='Data')

    latency_model, ISI12_model = get_latency_and_ISI12(cell)
    ax.plot(latency_model, ISI12_model, 'o', color=color_model, alpha=0.5, label='Model')

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('$ISI_{1/2}$ (ms)')
    ax.text(-0.25, 1.0, 'B', transform=ax.transAxes, size=18, weight='bold')

    # f-I curve
    ax = pl.Subplot(fig, outer[2, 0])
    fig.add_subplot(ax)

    with open(os.path.join(save_dir_model, model, 'img', 'IV', 'fi_curve', 'fi_dict.json'), 'r') as f:
        fi_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'IV', 'fi_curve', 'rat', exp_cell, 'fi_dict.json'), 'r') as f:
        fi_dict_data = json.load(f)

    plot_fi_curve_on_ax(ax, color_line=color_model, **fi_dict_model)
    plot_fi_curve_on_ax(ax, color_line=color_exp, **fi_dict_data)

    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.25, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')
    # np.max(np.diff(fi_dict_model['firing_rates']))

    # fit fI-curve
    FI_a = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'FI_a.npy'))
    FI_b = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'FI_b.npy'))
    FI_c = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'FI_c.npy'))
    RMSE = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'RMSE.npy'))
    #print 'RMSE over cells: ', np.min(RMSE), np.max(RMSE)

    amps_greater0, firing_rates_model = simulate_and_compute_fI_curve(cell)
    FI_a_model, FI_b_model, FI_c_model, RMSE_model = fit_fI_curve(amps_greater0, firing_rates_model)
    #print 'RMSE model: ', RMSE_model

    ax = pl.subplot(outer[0, 1])
    ax.plot(FI_a, FI_b, 'o', color=color_exp, alpha=0.5)
    ax.plot([FI_a_model], [FI_b_model], 'o', color=color_model, alpha=0.5)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.text(-0.25, 1.0, 'D', transform=ax.transAxes, size=18, weight='bold')

    ax = pl.subplot(outer[1, 1])
    ax.plot(FI_b, FI_c, 'o', color=color_exp, alpha=0.5)
    ax.plot([FI_b_model], [FI_c_model], 'o', color=color_model, alpha=0.5)
    ax.set_xlabel('b')
    ax.set_ylabel('c')

    ax = pl.subplot(outer[2, 1])
    ax.plot(FI_c, FI_a, 'o', color=color_exp, alpha=0.5)
    ax.plot([FI_c_model], [FI_a_model], 'o', color=color_model, alpha=0.5)
    ax.set_xlabel('c')
    ax.set_ylabel('a')

    pl.tight_layout()
    pl.subplots_adjust(right=0.95)
    pl.savefig(os.path.join(save_dir_img, 'reproduction_firing.png'))
    pl.show()