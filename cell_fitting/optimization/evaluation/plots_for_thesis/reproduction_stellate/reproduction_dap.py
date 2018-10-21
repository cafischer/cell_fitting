import numpy as np
import os
import json
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.gridspec as gridspec
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model
from cell_fitting.optimization.evaluation.plot_double_ramp import plot_current_threshold_on_ax
from cell_fitting.data.plot_doubleramp import get_inj_doubleramp_params, get_i_inj_double_ramp_full
from cell_fitting.optimization.evaluation.plot_double_ramp.plot_doubleramp_summary import plot_current_threshold_all_cells_on_ax
from cell_fitting.optimization.evaluation.plot_rampIV import simulate_rampIV
from nrn_wrapper import Cell
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_fitting.util import characteristics_dict_for_plotting
pl.style.use('paper_subplots')


# TODO: check simulation_params (e.g. dt)
# TODO: check all exp. data are v_shifted
if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    exp_cell_dr = '2015_08_06d'
    v_init = -75
    color_model = '0.5'
    ramp_amp = 3.5

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'), mechanism_dir)

    fig = pl.figure(figsize=(12, 4.5))
    outer = gridspec.GridSpec(2, 4, width_ratios=[3, 1, 1, 1.5])

    # Explanation DAP characteristics
    ax = pl.Subplot(fig, outer[:, 0])
    fig.add_subplot(ax)

    v_exp, t_exp, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'rampIV', ramp_amp)

    characteristics = ['AP_max_idx', 'fAHP_min_idx', 'DAP_max_idx', 'DAP_width_idx']
    start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
    v_rest = np.mean(v_exp[0:start_i_inj])
    characteristics_exp = np.array(get_spike_characteristics(v_exp, t_exp, characteristics, v_rest,
                                                             std_idx_times=(0, 1), check=False,
                                                             **get_spike_characteristics_dict(for_data=True)),
                                   dtype=int)

    t_exp -= 8
    ax.plot(t_exp, v_exp, 'k')
    ax.annotate('', xy=(t_exp[characteristics_exp[0]], v_exp[characteristics_exp[0]]),
                xytext=(t_exp[characteristics_exp[2]], v_exp[characteristics_exp[0]]),
                arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0})
    ax.annotate('DAP time', xy=((t_exp[characteristics_exp[0]] + t_exp[characteristics_exp[2]])/2.0,
                                v_exp[characteristics_exp[0]] + 0.5),
                verticalalignment='bottom', horizontalalignment='left')
    ax.plot([t_exp[characteristics_exp[2]], t_exp[characteristics_exp[2]]],
            [v_exp[characteristics_exp[2]], v_exp[characteristics_exp[0]]],
            '--', color='0.5')  # helper line DAP max vertical
    ax.set_xlim(0, 145)
    ax.set_xlabel('Mem. pot. (mV)')
    ax.set_ylabel('Time (ms)')

    axins = inset_axes(ax, width='60%', height='60%', loc=1)
    axins.plot(t_exp, v_exp, 'k')
    axins.annotate('', xy=(t_exp[characteristics_exp[2]], v_exp[characteristics_exp[1]]),
                   xytext=(t_exp[characteristics_exp[2]], v_exp[characteristics_exp[2]]),
                   arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0})
    axins.annotate('DAP deflection', xy=(t_exp[characteristics_exp[2]] + 1.0, v_exp[characteristics_exp[2]]),
                   verticalalignment='bottom', horizontalalignment='left')
    axins.annotate('', xy=(t_exp[characteristics_exp[2]], v_rest),
                   xytext=(t_exp[characteristics_exp[2]], v_exp[characteristics_exp[2]]),
                   arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0})
    axins.annotate('DAP amp.', xy=(t_exp[characteristics_exp[2]] + 1.0, v_rest),
                   verticalalignment='bottom', horizontalalignment='left')
    halfmax = v_rest + (v_exp[characteristics_exp[1]] - v_rest) / 2.0
    axins.annotate('', xy=(t_exp[characteristics_exp[1]], halfmax),
                   xytext=(t_exp[characteristics_exp[3]], halfmax),
                   arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0})
    axins.annotate('DAP width', xy=((t_exp[characteristics_exp[1]]+t_exp[characteristics_exp[3]])/2.0, halfmax),
                   verticalalignment='bottom', horizontalalignment='center')
    axins.plot([t_exp[characteristics_exp[1]], t_exp[characteristics_exp[1]]], [v_rest, v_exp[characteristics_exp[1]]],
               '--', color='0.5')  # helper line fAHP vertical
    axins.plot([t_exp[characteristics_exp[1]], 21-8],
               [v_exp[characteristics_exp[1]], v_exp[characteristics_exp[1]]],
               '--', color='0.5')  # helper line fAHP horizontal
    axins.set_ylim(v_rest, -50)
    axins.set_xlim(10-8, 35-8)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # DAP
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[:, 3], hspace=0.1, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    v_exp, t_exp, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'rampIV', ramp_amp)
    v_model, t_model, _ = simulate_model(cell, 'rampIV', ramp_amp, t_exp[-1], v_init=v_init)

    ax0.plot(t_exp, v_exp, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, color_model, label='Model')
    ax1.plot(t_exp, i_inj, 'k')

    ax0.legend()
    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])

    # distribution of DAP characteristics
    axes = [outer[0, 1], outer[0, 2], outer[1, 1], outer[1, 2]]
    characteristics = ['DAP_deflection', 'DAP_amp', 'DAP_time', 'DAP_width']
    characteristics_dict_plot = characteristics_dict_for_plotting()

    v, t, i_inj = simulate_rampIV(cell, ramp_amp, v_init=-75)
    start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
    v_rest = np.mean(v[0:start_i_inj])
    characteristics_mat_model = np.array(get_spike_characteristics(v, t, characteristics, v_rest, check=False,
                                                                   **get_spike_characteristics_dict()),
                                           dtype=float)

    characteristics_mat_exp = np.load(os.path.join(save_dir_data_plots, 'spike_characteristics/distributions/rat',
                                                   'characteristics_mat.npy')).astype(float)
    characteristics_exp = np.load(os.path.join(save_dir_data_plots, 'spike_characteristics/distributions/rat',
                                               'return_characteristics.npy'))

    for characteristic_idx, characteristic in enumerate(characteristics):
        ax = pl.Subplot(fig, axes[characteristic_idx])
        fig.add_subplot(ax)

        characteristic_idx_exp = np.where(characteristic == characteristics_exp)[0][0]
        not_nan_exp = ~np.isnan(characteristics_mat_exp[:, characteristic_idx_exp])
        ax.hist(characteristics_mat_exp[:, characteristic_idx_exp][not_nan_exp], bins=100, color='k')
        ax.axvline(characteristics_mat_model[characteristic_idx], 0, 1, color=color_model, linewidth=2.0)

        ax.set_xlabel(characteristics_dict_plot[characteristic])
        ax.set_ylabel('Frequency')


    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'reproduction_dap.png'))
    pl.show()