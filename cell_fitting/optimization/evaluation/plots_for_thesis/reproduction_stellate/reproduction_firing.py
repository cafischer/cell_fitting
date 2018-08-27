import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.plot_IV import plot_fi_curve_on_ax
pl.style.use('paper_subplots')


# TODO: colors
# TODO: check simulation_params (e.g. dt)
# TODO: check all exp. data are v_shifted
# TODO: check sag_amps and v_deflection data
if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    v_init = -75
    color_model = '0.5'
    step_amp = 0.4

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'), mechanism_dir)

    fig = pl.figure(figsize=(8, 6))
    outer = gridspec.GridSpec(2, 2)

    # pos. step
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.1, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'IV', step_amp)
    v_model, t_model, _ = simulate_model(cell, 'IV', step_amp, t_data[-1], v_init=v_init)

    ax0.plot(t_data, v_data, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, color_model, label='Model')
    ax1.plot(t_data, i_inj, 'k')

    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.15, 0.5)
    ax1.get_yaxis().set_label_coords(-0.15, 0.5)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])
    ax0.legend()

    # f-I curve
    ax = pl.Subplot(fig, outer[0, 1])
    fig.add_subplot(ax)

    with open(os.path.join(save_dir_model, model, 'img', 'IV', 'fi_curve', 'fi_dict.json'), 'r') as f:
        fi_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'IV', 'fi_curve', 'rat', exp_cell, 'fi_dict.json'), 'r') as f:
        fi_dict_data = json.load(f)

    plot_fi_curve_on_ax(ax, color_line=color_model, **fi_dict_model)
    plot_fi_curve_on_ax(ax, color_line='k', **fi_dict_data)

    ax.get_yaxis().set_label_coords(-0.15, 0.5)


    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'reproduction_firing.png'))
    pl.show()