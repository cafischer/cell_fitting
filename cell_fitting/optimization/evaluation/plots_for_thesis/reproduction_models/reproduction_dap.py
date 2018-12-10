import numpy as np
import os
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model, get_spike_characteristics_dict
from cell_fitting.optimization.evaluation.plot_rampIV import simulate_rampIV
from cell_fitting.optimization.errfuns import rms
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.util import characteristics_dict_for_plotting
from cell_characteristics.analyze_APs import get_spike_characteristics
pl.style.use('paper_subplots')


def curly_bracket(ax, pos=(0, 0), scalex=1, scaley=1, text="", textkw=None, linekw=None):
    if textkw is None:
        textkw = {}
    if linekw is None:
        linekw = {}
    x = np.array([0, 0.05, 0.45, 0.5])
    y = np.array([0,-0.01,-0.01,-0.02])
    x = np.concatenate((x, x+0.5))
    y = np.concatenate((y, y[::-1]))
    ax.plot(x*scalex+pos[0], y*scaley+pos[1], clip_on=False,
            transform=ax.get_xaxis_transform(), **linekw)
    ax.text(pos[0]+0.5*scalex, (y.min()-0.01)*scaley+pos[1], text,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", **textkw)


def rectangle_with_text(ax, x_m, y_m, width, height, text, fc='0.5'):
    x = x_m - width / 2.
    y = y_m - height / 2.
    rect = pl.Rectangle((x, y), width=width, height=height,
                        fill=True, facecolor=fc, clip_on=False)
    ax.add_patch(rect)
    ax.text(x_m, y_m - height, text, ha="center", va="center", color='k')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    models = ['2', '3', '4', '5', '6']
    exp_cell = '2015_08_26b'
    exp_cell_dr = '2015_08_06d'
    color_exp = '#0099cc'
    color_model = 'k'
    ramp_amp = 3.5
    standard_sim_params = get_standard_simulation_params()
    load_mechanism_dir(mechanism_dir)

    # plot 1: DAP
    fig = pl.figure(figsize=(12, 4.0))
    outer = gridspec.GridSpec(1, 5, wspace=0.15)

    for model_idx, model in enumerate(models):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, model_idx], hspace=0.1,
                                                 height_ratios=[5, 1])
        ax0 = pl.Subplot(fig, inner[0])
        ax1 = pl.Subplot(fig, inner[1])
        fig.add_subplot(ax0)
        fig.add_subplot(ax1)

        v_exp, t_exp, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'rampIV', ramp_amp)
        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))  # TODO: cell_rounded
        v_model, t_model, _ = simulate_model(cell, 'rampIV', ramp_amp, t_exp[-1], **standard_sim_params)

        fAHP_min_idx = get_spike_characteristics(v_model, t_model, ['fAHP_min_idx'], v_model[0], check=False,
                                                 **get_spike_characteristics_dict(for_data=False))[0]
        #print 'RMSE: ', rms(v_exp, v_model)
        #print 'RMSE: ', rms(v_exp[fAHP_min_idx:], v_model[fAHP_min_idx:])

        start_i_inj = np.where(i_inj)[0][0]
        vrest_data = np.mean(v_exp[:start_i_inj])
        vrest_model = np.mean(v_model[:start_i_inj])

        ax0.plot(t_exp, v_exp-vrest_data, color_exp, label='Data')
        ax0.plot(t_model, v_model-vrest_model, color_model, label='Model')
        ax1.plot(t_exp, i_inj, 'k')

        ax0.set_xticks([])
        ax0.set_ylim(-5, 140)
        ax1.set_xlabel('Time (ms)')
        if model_idx == 0:
            ax0.legend()
            ax0.set_ylabel('Mem. pot. amp. (mV)')
            ax1.set_ylabel('Current (nA)')
            ax0.get_yaxis().set_label_coords(-0.3, 0.5)
            ax1.get_yaxis().set_label_coords(-0.3, 0.5)
            ax1.set_yticks([np.min(i_inj), np.max(i_inj)])

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'reproduction_dap1_models.png'))

    # plot 2
    fig = pl.figure(figsize=(8, 6))
    outer = gridspec.GridSpec(2, 2)

    # distribution of DAP characteristics
    axes = [outer[0, 0], outer[0, 1], outer[1, 0], outer[1, 1]]
    characteristics = ['DAP_deflection', 'DAP_amp', 'DAP_time', 'DAP_width']
    characteristics_dict_plot = characteristics_dict_for_plotting()

    characteristics_mat_exp = np.load(os.path.join(save_dir_data_plots, 'spike_characteristics/rat',
                                                   'characteristics_mat.npy')).astype(float)
    characteristics_exp = np.load(os.path.join(save_dir_data_plots, 'spike_characteristics/rat',
                                               'return_characteristics.npy'))

    for characteristic_idx, characteristic in enumerate(characteristics):
        ax = pl.Subplot(fig, axes[characteristic_idx])
        fig.add_subplot(ax)

        characteristic_idx_exp = np.where(characteristic == characteristics_exp)[0][0]
        not_nan_exp = ~np.isnan(characteristics_mat_exp[:, characteristic_idx_exp])
        ax.hist(characteristics_mat_exp[:, characteristic_idx_exp][not_nan_exp], bins=100, color=color_exp,
                label='Data')

        for model_idx, model in enumerate(models):
            cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))  # TODO: cell_rounded
            v, t, i_inj = simulate_rampIV(cell, ramp_amp, v_init=-75)
            start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
            v_rest = np.mean(v[0:start_i_inj])
            characteristics_mat_model = np.array(get_spike_characteristics(v, t, characteristics, v_rest, check=False,
                                                                           **get_spike_characteristics_dict()),
                                                 dtype=float)
            ax.axvline(characteristics_mat_model[characteristic_idx], 0, 1, color=color_model, linewidth=1.0, label='Model')
            ax.annotate(str(model_idx+1),
                        xy=(characteristics_mat_model[characteristic_idx], ax.get_ylim()[1] + 0.01),
                        color=color_model, fontsize=8, ha='center')

        ax.set_xlabel(characteristics_dict_plot[characteristic])
        ax.set_ylabel('Frequency')
        ax.get_yaxis().set_label_coords(-0.15, 0.5)

        if characteristic_idx == 0:
            # legend
            custom_lines = [Line2D([0], [0], color=color_exp, lw=1.0),
                            Line2D([0], [0], color=color_model, lw=1.0)]
            ax.legend(custom_lines, ['Data', 'Model'])

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'reproduction_dap2_models.png'))
    pl.show()