import numpy as np
import os
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from nrn_wrapper import Cell
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model, get_spike_characteristics_dict
from cell_fitting.optimization.evaluation.plot_rampIV import simulate_rampIV
from cell_fitting.optimization.errfuns import rms
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.util import characteristics_dict_for_plotting, change_color_brightness
from matplotlib.colors import to_rgb
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_characteristics import to_idx
pl.style.use('paper')


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
    save_dir_img = '/home/cfischer/Dropbox/thesis/figures_results_paper'
    save_dir_model = '/home/cfischer/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cfischer/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cfischer/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cfischer/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    exp_cell_dr = '2015_08_06d'
    color_exp = '#0099cc'
    color_model = 'k'
    ramp_amp = 3.5
    standard_sim_params = get_standard_simulation_params()
    units = ['mV', 'mV', 'ms', 'ms']

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'), mechanism_dir)

    # plot 1: DAP
    fig = pl.figure(figsize=(6, 4.5))
    outer = gridspec.GridSpec(1, 1)
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.1, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    v_exp, t_exp, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'rampIV', ramp_amp)
    v_model, t_model, _ = simulate_model(cell, 'rampIV', ramp_amp, t_exp[-1], **standard_sim_params)

    fAHP_min_idx, DAP_width_idx = get_spike_characteristics(v_model, t_model, ['fAHP_min_idx', 'DAP_width_idx'],
                                                            v_model[0], check=False,
                                                            **get_spike_characteristics_dict(for_data=False))
    print 'RMSE (0 ms-fAHP): %.2f' % rms(v_exp[:fAHP_min_idx], v_model[:fAHP_min_idx])
    print 'RMSE (fAHP-DAP width): %.2f' % rms(v_exp[fAHP_min_idx:DAP_width_idx], v_model[fAHP_min_idx:DAP_width_idx])
    print 'RMSE (DAP width-150 ms): %.2f' % rms(v_exp[DAP_width_idx:to_idx(150, standard_sim_params['dt'])],
                         v_model[DAP_width_idx:to_idx(150, standard_sim_params['dt'])])

    start_i_inj = np.where(i_inj)[0][0]
    vrest_data = np.mean(v_exp[:start_i_inj])
    vrest_model = np.mean(v_model[:start_i_inj])

    # ax0.plot(t_exp, v_exp, color_exp, label='Data')
    # ax0.plot(t_model, v_model, color_model, label='Model')
    ax0.plot(t_exp, v_exp-vrest_data, color_exp, label='Data')
    ax0.plot(t_model, v_model-vrest_model, color_model, label='Model')
    #ax0.plot(t_model[fAHP_min_idx:], v_model[fAHP_min_idx:], color='y')
    ax1.plot(t_exp, i_inj, 'k')

    ax0.set_xlim(0, t_exp[-1])
    ax1.set_xlim(0, t_exp[-1])
    ax0.legend()
    ax0.set_xticks([])
    #ax0.set_ylabel('Mem. pot. $-V_{rest}$ (mV)')
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.1, 0.5)
    ax1.get_yaxis().set_label_coords(-0.1, 0.5)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])

    pl.tight_layout()
    #pl.savefig(os.path.join(save_dir_img, 'reproduction_dap1.png'))

    # plot 2
    fig = pl.figure(figsize=(10, 4.5))
    outer = gridspec.GridSpec(2, 3, width_ratios=[3, 1, 1])

    # Explanation DAP characteristics
    ax = pl.Subplot(fig, outer[:, 0])
    fig.add_subplot(ax)

    v_exp, t_exp, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'rampIV', ramp_amp)

    characteristics = ['AP_max_idx', 'fAHP_min_idx', 'DAP_max_idx', 'DAP_width_idx']
    start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
    v_rest = np.mean(v_exp[0:start_i_inj])
    characteristics_exp_cells = np.array(get_spike_characteristics(v_exp, t_exp, characteristics, v_rest,
                                                                   std_idx_times=(0, 1), check=False,
                                                                   **get_spike_characteristics_dict(for_data=True)),
                                         dtype=int)
    t_shift = -8
    t_exp += t_shift
    ax.plot(t_exp, v_exp, 'k')
    ax.annotate('', xy=(t_exp[characteristics_exp_cells[0]], v_exp[characteristics_exp_cells[0]]),
                xytext=(t_exp[characteristics_exp_cells[2]], v_exp[characteristics_exp_cells[0]]),
                arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0})
    ax.annotate('$\mathrm{Time_{AP-DAP}}$', xy=((t_exp[characteristics_exp_cells[0]] + t_exp[characteristics_exp_cells[2]]) / 2.0,
                                       v_exp[characteristics_exp_cells[0]] + 0.5),
                verticalalignment='bottom', horizontalalignment='left')
    ax.plot([t_exp[characteristics_exp_cells[2]], t_exp[characteristics_exp_cells[2]]],
            [v_exp[characteristics_exp_cells[2]], v_exp[characteristics_exp_cells[0]]],
            '--', color='0.5')  # helper line DAP max vertical
    ax.set_xlim(0, 145)
    ax.set_ylabel('Mem. pot. (mV)')
    ax.set_xlabel('Time (ms)')

    axins = inset_axes(ax, width='60%', height='60%', loc=1)
    axins.plot(t_exp, v_exp, 'k')
    ax.annotate('mAHP', xy=(63., -73), verticalalignment='top', horizontalalignment='center')
    axins.annotate('fAHP', xy=(t_exp[characteristics_exp_cells[1]], v_exp[characteristics_exp_cells[1]] - 0.5),
                   verticalalignment='top', horizontalalignment='center')
    axins.annotate('', xy=(21+t_shift, v_exp[characteristics_exp_cells[1]]),
                   xytext=(21+t_shift, v_exp[characteristics_exp_cells[2]]),
                   arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0})
    axins.annotate('DAP deflection', xy=(21 + t_shift + 1.0,
                                         (v_exp[characteristics_exp_cells[2]] + v_exp[characteristics_exp_cells[1]]) / 2.),
                   verticalalignment='center', horizontalalignment='left')
    axins.annotate('', xy=(t_exp[characteristics_exp_cells[2]], v_rest),
                   xytext=(t_exp[characteristics_exp_cells[2]], v_exp[characteristics_exp_cells[2]]),
                   arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0})
    axins.annotate('DAP amp.', xy=(t_exp[characteristics_exp_cells[2]] + 1.0, v_rest),
                   verticalalignment='bottom', horizontalalignment='left')
    halfmax = v_rest + (v_exp[characteristics_exp_cells[1]] - v_rest) / 2.0
    axins.annotate('', xy=(t_exp[characteristics_exp_cells[1]], halfmax),
                   xytext=(t_exp[characteristics_exp_cells[3]], halfmax),
                   arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0})
    axins.annotate('DAP width', xy=((t_exp[characteristics_exp_cells[1]] + t_exp[characteristics_exp_cells[3]]) / 2.0, halfmax),
                   verticalalignment='bottom', horizontalalignment='center')
    axins.plot([t_exp[characteristics_exp_cells[1]], t_exp[characteristics_exp_cells[1]]], [v_rest, v_exp[characteristics_exp_cells[1]]],
               '--', color='0.5')  # helper line fAHP vertical
    axins.plot([t_exp[characteristics_exp_cells[1]], 21 + t_shift],
               [v_exp[characteristics_exp_cells[1]], v_exp[characteristics_exp_cells[1]]],
               '--', color='0.5')  # helper line fAHP horizontal
    axins.plot([t_exp[characteristics_exp_cells[2]], 21 + t_shift],
               [v_exp[characteristics_exp_cells[2]], v_exp[characteristics_exp_cells[2]]],
               '--', color='0.5')  # helper line DAP max horizontal
    axins.axhline(v_rest, linestyle='--', color='0.5')  # helper line v_rest

    # inset
    axins.set_ylim(v_rest - 2, -50)
    axins.set_xlim(8.5+t_shift, 35+t_shift)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # letter
    ax.text(-0.16, 1.0, 'A', transform=ax.transAxes, size=18, weight='bold')

    # distribution of DAP characteristics
    axes = [outer[0, 1], outer[0, 2], outer[1, 1], outer[1, 2]]
    characteristics = ['DAP_deflection', 'DAP_amp', 'DAP_time', 'DAP_width']
    characteristics_dict_plot = characteristics_dict_for_plotting()

    v, t, i_inj = simulate_rampIV(cell, ramp_amp, v_init=-75)
    start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
    v_rest = np.mean(v[0:start_i_inj])
    v_rest_exp = np.mean(v_exp[0:start_i_inj])
    std_idx_times = (0, min(1, start_i_inj * (t_exp[1]-t_exp[0])))
    characteristics_mat_model = np.array(get_spike_characteristics(v, t, characteristics, v_rest, check=False,
                                                                   **get_spike_characteristics_dict()), dtype=float)

    characteristics_mat_exp_cells = np.load(os.path.join(save_dir_data_plots, 'spike_characteristics/rat',
                                                   'characteristics_mat.npy'), allow_pickle=True).astype(float)
    characteristics_exp_cells = np.load(os.path.join(save_dir_data_plots, 'spike_characteristics/rat',
                                        'return_characteristics.npy'))
    cell_ids_characteristics = np.load(os.path.join(save_dir_data_plots, 'spike_characteristics/rat',
                                                    'cell_ids.npy'))

    for characteristic_idx, characteristic in enumerate(characteristics):
        ax = pl.Subplot(fig, axes[characteristic_idx])
        fig.add_subplot(ax)

        characteristic_idx_exp = np.where(characteristic == characteristics_exp_cells)[0][0]
        not_nan_exp = ~np.isnan(characteristics_mat_exp_cells[:, characteristic_idx_exp])
        ax.hist(characteristics_mat_exp_cells[:, characteristic_idx_exp][not_nan_exp], bins=100, color=color_exp,
                label='Data')
        ax.axvline(characteristics_mat_model[characteristic_idx], 0, 1, color=color_model, linewidth=1.3, label='Model')
        ax.axvline(characteristics_mat_exp_cells[cell_ids_characteristics==exp_cell, characteristic_idx_exp], 0, 1,
                   color='m', linewidth=1.3)

        # inside std
        std = np.std(characteristics_mat_exp_cells[:, characteristic_idx_exp][not_nan_exp])
        mean = np.mean(characteristics_mat_exp_cells[:, characteristic_idx_exp][not_nan_exp])
        print mean-std, mean+std
        print characteristics_mat_model[characteristic_idx]

        ax.set_xlabel(characteristics_dict_plot[characteristic] + ' ('+units[characteristic_idx]+')')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, None)

        if characteristic_idx == 0:
            # legend
            custom_lines = [Line2D([0], [0], color=color_exp, lw=1.0),
                            Line2D([0], [0], color=color_model, lw=1.0),
                            Line2D([0], [0], color='m', lw=1.0)]
            legend = ax.legend(custom_lines, ['Data', 'Model', 'Target \ncell'])

            # letter
            ax.text(-0.37, 1.0, 'B', transform=ax.transAxes, size=18, weight='bold')

    pl.tight_layout()
    pl.subplots_adjust(top=0.95, bottom=0.12)
    pl.savefig(os.path.join(save_dir_img, 'reproduction_dap2.png'))
    #pl.show()

    # 2d scatter plots
    fig, axes = pl.subplots(4, 4, figsize=(9, 9))

    dtick = {'DAP_deflection': 2.5, 'DAP_amp': 10, 'DAP_width': 15, 'DAP_time': 2.5}

    for i, characteristic1 in enumerate(characteristics):
        for j, characteristic2 in enumerate(characteristics):
            ax = axes[i, j]

            characteristic_idx1 = np.where(characteristic1 == characteristics_exp_cells)[0][0]
            characteristic_idx2 = np.where(characteristic2 == characteristics_exp_cells)[0][0]
            not_nan_exp = np.logical_and(~np.isnan(characteristics_mat_exp_cells[:, characteristic_idx1]),
                                         ~np.isnan(characteristics_mat_exp_cells[:, characteristic_idx2]))
            ax.scatter(characteristics_mat_exp_cells[:, characteristic_idx1][not_nan_exp],
                       characteristics_mat_exp_cells[:, characteristic_idx2][not_nan_exp], color=color_exp,
                       label='Data', alpha=0.5)
            ax.scatter(characteristics_mat_model[i], characteristics_mat_model[j], color=color_model, label='Model',
                       alpha=0.5)
            ax.scatter(characteristics_mat_exp_cells[cell_ids_characteristics==exp_cell, characteristic_idx1],
                       characteristics_mat_exp_cells[cell_ids_characteristics==exp_cell, characteristic_idx2],
                       color='m', alpha=0.8)

            ax.set_xticks(np.arange(0, np.max(characteristics_mat_exp_cells[:, characteristic_idx1][not_nan_exp]),
                                    dtick[characteristic1]))
            ax.set_yticks(np.arange(0, np.max(characteristics_mat_exp_cells[:, characteristic_idx2][not_nan_exp]),
                                    dtick[characteristic2]))

            # characteristics = ['DAP_deflection', 'DAP_amp', 'DAP_time', 'DAP_width']
            max_characteristics = [11., 35., 10., 68.]
            ax.set_xlim(0, max_characteristics[i])
            ax.set_ylim(0, max_characteristics[j])
            ax.set_xlabel(characteristics_dict_plot[characteristic1] + ' ('+units[i]+')')
            ax.set_ylabel(characteristics_dict_plot[characteristic2] + ' ('+units[j]+')')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'reproduction_dap2_scatter.png'))
    pl.show()