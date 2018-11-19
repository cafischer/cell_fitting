import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from nrn_wrapper import Cell
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import plot_sag_vs_steady_state_on_ax
from cell_fitting.optimization.evaluation.plot_blocking.block_channel import block_channel, plot_channel_block_on_ax
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import compute_v_sag_and_steady_state
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.util import get_channel_color_for_plotting
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
    step_amp = -0.1
    standard_sim_params = get_standard_simulation_params()

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'), mechanism_dir)

    fig = pl.figure(figsize=(8, 6))
    outer = gridspec.GridSpec(2, 2)

    # sag: mem. pot.
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.15, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'IV', step_amp)
    v_model, t_model, i_inj_model = simulate_model(cell, 'IV', step_amp, t_data[-1], **standard_sim_params)

    start_i_inj = np.where(i_inj)[0][0]
    vrest_data = np.mean(v_data[:start_i_inj])
    vrest_model = np.mean(v_model[:start_i_inj])

    # ax0.plot(t_data, v_data, color_exp, label='Data')
    # ax0.plot(t_model, v_model, color_model, label='Model')
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
    # letter
    ax0.text(-0.25, 1.0, 'A', transform=ax0.transAxes, size=18, weight='bold')

    # sag vs. steady-state
    ax = pl.Subplot(fig, outer[0, 1])
    fig.add_subplot(ax)

    with open(os.path.join(save_dir_model, model, 'img', 'IV', 'sag', 'sag_dict.json'), 'r') as f:
        sag_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'IV', 'sag', exp_cell, 'sag_dict.json'), 'r') as f:
        sag_dict_data = json.load(f)

    plot_sag_vs_steady_state_on_ax(ax, color_lines=color_model, label=False, **sag_dict_model)
    plot_sag_vs_steady_state_on_ax(ax, color_lines=color_exp, label=False, **sag_dict_data)

    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    custom_lines = [Line2D([0], [0], marker='s', color='None', markerfacecolor='0.5', markeredgecolor='0.5', lw=1.0),
                    Line2D([0], [0], marker='$\cup$', color='None', markerfacecolor='0.5', markeredgecolor='0.5', lw=1.0)]
    ax.legend(custom_lines, ['Steady state', 'Sag'], loc='upper left')
    ax.text(-0.25, 1.0, 'B', transform=ax.transAxes, size=18, weight='bold')

    # block HCN
    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)

    percent_block = 100
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))
    block_channel(cell, 'hcn_slow', percent_block)
    v_after_block, _, _ = simulate_model(cell, 'IV', step_amp, t_data[-1], **standard_sim_params)

    # plot_channel_block_on_ax(ax, ['hcn_slow'], t_model, v_model, np.array([v_after_block]), percent_block,
    #                          color=color_model)
    # ax.set_ylim(-85, -70)
    vrest_after_block = np.mean(v_after_block[:start_i_inj])
    plot_channel_block_on_ax(ax, ['hcn_slow'], t_model, v_model - vrest_model,
                             np.array([v_after_block - vrest_after_block]), percent_block,
                             color=color_model)  # TODO: maybe shift in rest also interesting to see?!
    ax.set_ylim(-4, 2.5)
    custom_lines = [Line2D([0], [0], marker='o', color='k', lw=1.0),
                    Line2D([0], [0], marker='o', color=get_channel_color_for_plotting()['hcn_slow'], lw=1.0)]
    ax.legend(custom_lines, ['Without block (model)', '100% block of HCN (model)'], loc='upper right')
    ax.text(-0.25, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')

    # sag amp. vs voltage deflection for all cells
    ax = pl.Subplot(fig, outer[1, 1])
    fig.add_subplot(ax)

    sag_amps_data = np.load(os.path.join(save_dir_data_plots, 'IV', 'sag', 'rat', str(step_amp),
                                         'sag_amps.npy'))
    v_deflections_data = np.load(os.path.join(save_dir_data_plots, 'IV', 'sag', 'rat', str(step_amp),
                                              'v_deflections.npy'))
    ax.plot(sag_amps_data, v_deflections_data, 'o', color=color_exp, alpha=0.5, label='Data')

    start_step_idx = np.nonzero(i_inj_model)[0][0]
    end_step_idx = np.nonzero(i_inj_model)[0][-1] + 1
    v_sags, v_steady_states, _ = compute_v_sag_and_steady_state([v_model], [step_amp], AP_threshold=0,
                                                                start_step_idx=start_step_idx,
                                                                end_step_idx=end_step_idx)
    sag_amp_model = v_steady_states[0] - v_sags[0]
    vrest = np.mean(v_model[:start_step_idx])
    v_deflection_model = vrest - v_steady_states[0]
    ax.plot(sag_amp_model, v_deflection_model, 'o', color=color_model, alpha=0.5, label='Model')

    ax.set_xlabel('Sag deflection')
    ax.set_ylabel('Amp. at steady state')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    #ax.legend()
    ax.text(-0.25, 1.0, 'D', transform=ax.transAxes, size=18, weight='bold')

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'reproduction_sag.png'))
    pl.show()