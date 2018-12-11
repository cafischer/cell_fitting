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
    outer = gridspec.GridSpec(1, 5, wspace=0.2)

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
            ax0.get_yaxis().set_label_coords(-0.25, 0.5)
            ax1.get_yaxis().set_label_coords(-0.25, 0.5)
            ax1.set_yticks([np.min(i_inj), np.max(i_inj)])

    pl.tight_layout()
    pl.subplots_adjust(top=0.97, bottom=0.12, left=0.06, right=0.99)
    pl.savefig(os.path.join(save_dir_img, 'reproduction_dap1_models.png'))