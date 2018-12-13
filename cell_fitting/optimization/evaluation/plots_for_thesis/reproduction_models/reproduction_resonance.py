import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from cell_fitting.read_heka import load_data, get_i_inj_standard_params
from cell_fitting.optimization.evaluation import simulate_model
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.evaluation.plot_zap import plot_impedance_on_ax
from cell_fitting.optimization.evaluation.plot_blocking.block_channel import block_channel, plot_channel_block_on_ax
from cell_fitting.optimization.evaluation.plot_zap import simulate_and_compute_zap_characteristics, \
    compute_res_freq_and_q_val
from matplotlib.lines import Line2D
from cell_fitting.util import get_channel_color_for_plotting
from cell_fitting.optimization.simulate import get_standard_simulation_params
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
    zap_amp = 0.1
    standard_sim_params = get_standard_simulation_params()
    load_mechanism_dir(mechanism_dir)

    # plot
    fig = pl.figure(figsize=(12, 9))
    outer = gridspec.GridSpec(3, 5)

    for model_idx, model in enumerate(models):

        # zap: mem. pot.
        zap_params = get_i_inj_standard_params('Zap20')
        zap_params['tstop'] = 34000 - standard_sim_params['dt']
        zap_params['dt'] = standard_sim_params['dt']
        zap_params['offset_dur'] = zap_params['onset_dur'] - standard_sim_params['dt']
        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'))
        if model == '4':
            zap_params['amp'] = 0.08
        v_model, t_model, i_inj_model, imp_smooth_model, frequencies_model, \
            res_freq_model, q_value_model = simulate_and_compute_zap_characteristics(cell, zap_params)

        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, model_idx], hspace=0.15,
                                                 height_ratios=[5, 1])
        ax0 = pl.Subplot(fig, inner[0])
        ax1 = pl.Subplot(fig, inner[1])
        fig.add_subplot(ax0)
        fig.add_subplot(ax1)

        v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'Zap20', zap_amp)

        start_i_inj = np.where(i_inj)[0][0]
        vrest_data = np.mean(v_data[:start_i_inj])
        vrest_model = np.mean(v_model[:start_i_inj])

        ax0.plot(t_data, v_data - vrest_data, color_exp, linewidth=0.3, label='Data')
        ax0.plot(t_model, v_model - vrest_model, color_model, linewidth=0.3, label='Model')
        ax1.plot(t_model, i_inj_model, linewidth=0.3, color='k')

        ax0.set_xticks([])
        ax1.set_yticks([np.min(i_inj), np.max(i_inj)])
        ax1.set_xlabel('Time (ms)')
        ax0.set_ylim(-6.5, 6.5)
        if model_idx == 0:
            ax0.set_ylabel('Mem. pot. (mV)')
            ax1.set_ylabel('Current (nA)')
            ax0.get_yaxis().set_label_coords(-0.25, 0.5)
            ax1.get_yaxis().set_label_coords(-0.25, 0.5)
            custom_lines = [Line2D([0], [0], color=color_exp, lw=1.0),
                            Line2D([0], [0], color=color_model, lw=1.0)]
            ax0.legend(custom_lines, ['Data', 'Model'], loc='upper right')
            ax0.text(-0.4, 1.0, 'A', transform=ax0.transAxes, size=18, weight='bold')

        # impedance
        ax = pl.Subplot(fig, outer[1, model_idx])
        fig.add_subplot(ax)

        with open(os.path.join(save_dir_data_plots, 'Zap20/rat', exp_cell, 'impedance_dict.json'), 'r') as f:
            impedance_dict_data = json.load(f)
        res_freq_data, q_value_data = compute_res_freq_and_q_val(np.array(impedance_dict_data['impedance']),
                                                                   np.array(impedance_dict_data['frequencies']))

        plot_impedance_on_ax(ax, frequencies_model, imp_smooth_model, color_line=color_model)
        plot_impedance_on_ax(ax, color_line=color_exp, **impedance_dict_data)

        ax.set_ylim(5, 60)
        if model_idx == 0:
            ax.get_yaxis().set_label_coords(-0.25, 0.5)
            ax.text(-0.4, 1.0, 'B', transform=ax.transAxes, size=18, weight='bold')
        else:
            ax.set_ylabel('')

        # block HCN
        ax = pl.Subplot(fig, outer[2, model_idx])
        fig.add_subplot(ax)

        percent_block = 100
        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'))
        block_channel(cell, 'hcn_slow', percent_block)
        v_after_block, _, _ = simulate_model(cell, 'Zap20', zap_amp, 34000 - standard_sim_params['dt'],
                                             **standard_sim_params)
        vrest_after_block = np.mean(v_after_block[:start_i_inj])
        plot_channel_block_on_ax(ax, ['hcn_slow'], t_model, v_model - vrest_model,
                                 np.array([v_after_block - vrest_after_block]), percent_block,
                                 color=color_model, label=False)
        ax.set_ylim(-6.5, 6.5)
        if model_idx == 0:
            ax.get_yaxis().set_label_coords(-0.25, 0.5)
            custom_lines = [Line2D([0], [0], marker='o', color='k', lw=1.0),
                            Line2D([0], [0], marker='o', color=get_channel_color_for_plotting()['hcn_slow'], lw=1.0)]
            ax.legend(custom_lines, ['Without block \n(model)', '100% block of \nHCN (model)'], loc='lower right')
            ax.text(-0.4, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')
        else:
            ax.set_ylabel('')

    pl.tight_layout()
    pl.subplots_adjust(top=0.97, bottom=0.06, right=0.99)
    pl.savefig(os.path.join(save_dir_img, 'reproduction_resonance_models.png'))
    pl.show()