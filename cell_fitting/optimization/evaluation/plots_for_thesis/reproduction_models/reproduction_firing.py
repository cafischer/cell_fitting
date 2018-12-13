import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model
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
    step_amp = 0.4
    standard_sim_params = get_standard_simulation_params()
    load_mechanism_dir(mechanism_dir)

    # plot
    fig = pl.figure(figsize=(12, 6))
    outer = gridspec.GridSpec(2, 5)

    # mem. pot.
    for model_idx, model in enumerate(models):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, model_idx], hspace=0.1,
                                                 height_ratios=[5, 1])
        ax0 = pl.Subplot(fig, inner[0])
        ax1 = pl.Subplot(fig, inner[1])
        fig.add_subplot(ax0)
        fig.add_subplot(ax1)

        v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'IV', step_amp)

        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'))
        v_model, t_model, _ = simulate_model(cell, 'IV', step_amp, t_data[-1], **standard_sim_params)

        start_i_inj = np.where(i_inj)[0][0]
        vrest_data = np.mean(v_data[:start_i_inj])
        vrest_model = np.mean(v_model[:start_i_inj])

        ax0.plot(t_data, v_data - vrest_data, color_exp, label='Data')
        ax0.plot(t_model, v_model - vrest_model, color_model, label='Model')
        ax1.plot(t_data, i_inj, 'k')

        ax0.set_xticks([])
        ax1.set_xlabel('Time (ms)')
        ax0.get_yaxis().set_label_coords(-0.25, 0.6)
        ax1.get_yaxis().set_label_coords(-0.25, 0.5)
        ax1.set_yticks([np.min(i_inj), np.max(i_inj)])
        ax0.set_ylim(-10, 125)
        if model_idx == 0:
            ax0.set_ylabel('Mem. pot. (mV)')
            ax1.set_ylabel('Current (nA)')
            ax0.legend()
            ax0.text(-0.4, 1.0, 'A', transform=ax0.transAxes, size=18, weight='bold')

    # F-I curve
    for model_idx, model in enumerate(models):
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, model_idx])
        ax = pl.Subplot(fig, inner[0])
        fig.add_subplot(ax)

        with open(os.path.join(save_dir_model, model, 'img', 'IV', 'fi_curve', 'fi_dict.json'), 'r') as f:
            fi_dict_model = json.load(f)

        with open(os.path.join(save_dir_data_plots, 'IV', 'fi_curve', 'rat', exp_cell, 'fi_dict.json'), 'r') as f:
            fi_dict_data = json.load(f)

        # plot_fi_curve_on_ax(ax, color_line=color_model, **fi_dict_model)
        # plot_fi_curve_on_ax(ax, color_line=color_exp, **fi_dict_data)
        ax.plot(fi_dict_data['amps'], fi_dict_data['firing_rates'], '-o', color=color_exp, markersize=4)
        ax.plot(fi_dict_model['amps'], fi_dict_model['firing_rates'], '-o', color=color_model, markersize=4)

        ax.set_xlabel('Current (nA)')
        ax.set_ylim(-3, 95)
        if model_idx == 0:
            ax.set_ylabel('Firing rate (Hz)')
            ax.get_yaxis().set_label_coords(-0.25, 0.5)
            ax.text(-0.4, 1.0, 'B', transform=ax.transAxes, size=18, weight='bold')

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'reproduction_firing_models.png'))
    pl.show()