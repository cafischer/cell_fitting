import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import plot_sag_vs_steady_state_on_ax
from cell_fitting.optimization.evaluation.plot_blocking.block_channel import block_channel, plot_channel_block_on_ax
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.util import get_channel_color_for_plotting
pl.style.use('paper_subplots')


if __name__ == '__main__':

    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    models = ['2', '3', '4', '5', '6']
    exp_cell = '2015_08_26b'
    color_exp = '#0099cc'
    color_model = 'k'
    step_amp = -0.1
    standard_sim_params = get_standard_simulation_params()
    load_mechanism_dir(mechanism_dir)

    # plot
    fig = pl.figure(figsize=(12, 9))
    outer = gridspec.GridSpec(3, 5)

    # sag: mem. pot.
    for model_idx, model in enumerate(models):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, model_idx], hspace=0.15,
                                                 height_ratios=[5, 1])
        ax0 = pl.Subplot(fig, inner[0])
        ax1 = pl.Subplot(fig, inner[1])
        fig.add_subplot(ax0)
        fig.add_subplot(ax1)

        v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'IV', step_amp)
        start_i_inj = np.where(i_inj)[0][0]
        vrest_data = np.mean(v_data[:start_i_inj])
        ax0.plot(t_data, v_data - vrest_data, color_exp, label='Data')
        ax1.plot(t_data, i_inj, 'k')

        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))  # TODO: cell_rounded
        v_model, t_model, i_inj_model = simulate_model(cell, 'IV', step_amp, t_data[-1], **standard_sim_params)
        vrest_model = np.mean(v_model[:start_i_inj])
        ax0.plot(t_model, v_model - vrest_model, color_model, label='Model')

        ax0.set_xticks([])
        ax1.set_xlabel('Time (ms)')
        ax1.set_yticks([np.min(i_inj), np.max(i_inj)])
        if model_idx == 0:
            ax0.set_ylabel('Mem. pot. (mV)')
            ax1.set_ylabel('Current (nA)')
            ax0.get_yaxis().set_label_coords(-0.25, 0.6)
            ax1.get_yaxis().set_label_coords(-0.25, 0.5)
            ax0.legend(loc='upper left')
            ax0.text(-0.4, 1.0, 'A', transform=ax0.transAxes, size=18, weight='bold')

    # sag vs. steady-state
    for model_idx, model in enumerate(models):

        ax = pl.Subplot(fig, outer[1, model_idx])
        fig.add_subplot(ax)

        with open(os.path.join(save_dir_data_plots, 'IV', 'sag', exp_cell, 'sag_dict.json'), 'r') as f:
            sag_dict_data = json.load(f)
        plot_sag_vs_steady_state_on_ax(ax, color_lines=color_exp, label=False, **sag_dict_data)

        with open(os.path.join(save_dir_model, model, 'img', 'IV', 'sag', 'sag_dict.json'), 'r') as f:
            sag_dict_model = json.load(f)
        plot_sag_vs_steady_state_on_ax(ax, color_lines=color_model, label=False, **sag_dict_model)

        ax.set_ylim(-87, -63)
        if model_idx == 0:
            ax.set_ylabel('Mem. pot. (mV)')
            ax.get_yaxis().set_label_coords(-0.25, 0.5)
            ax.text(-0.4, 1.0, 'B', transform=ax.transAxes, size=18, weight='bold')

            custom_lines = [
                Line2D([0], [0], marker='s', color='None', markerfacecolor='0.5', markeredgecolor='0.5', lw=1.0),
                Line2D([0], [0], marker='$\cup$', color='None', markerfacecolor='0.5', markeredgecolor='0.5',
                       lw=1.0)]
            ax.legend(custom_lines, ['Steady state', 'Sag'], loc='upper left')

    # blocking
    for model_idx, model in enumerate(models):
        ax = pl.Subplot(fig, outer[2, model_idx])
        fig.add_subplot(ax)

        percent_block = 100
        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))
        block_channel(cell, 'hcn_slow', percent_block)
        v_after_block, _, _ = simulate_model(cell, 'IV', step_amp, t_data[-1], **standard_sim_params)

        plot_channel_block_on_ax(ax, ['hcn_slow'], t_model, v_model, np.array([v_after_block]), percent_block,
                                 color=color_model, label=False)
        ax.set_ylim(-85, -74)
        ax.set_yticks(np.arange(-85, -71, 5))
        if model_idx == 0:
            ax.get_yaxis().set_label_coords(-0.25, 0.5)
            custom_lines = [Line2D([0], [0], marker='o', color='k', lw=1.0),
                            Line2D([0], [0], marker='o', color=get_channel_color_for_plotting()['hcn_slow'], lw=1.0)]
            ax.legend(custom_lines, ['Without block \n(model)', '100% block of \nHCN (model)'], loc='center left')
            ax.text(-0.4, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')
        else:
            ax.set_ylabel('')

    pl.tight_layout()
    pl.subplots_adjust(right=0.99, left=0.06, top=0.97, bottom=0.06)
    pl.savefig(os.path.join(save_dir_img, 'reproduction_sag_models.png'))
    pl.show()