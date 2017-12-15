import numpy as np
import os
import matplotlib.pyplot as pl
from scipy.stats import linregress, ttest_1samp
from cell_fitting.util import init_nan
pl.style.use('paper')

characteristic_name_dict = {'DAP_amp': 'DAP Amplitude', 'DAP_deflection': 'DAP Deflection', 'DAP_width': 'DAP Width',
                            'DAP_time': 'DAP Time', 'fAHP_amp': 'fAHP Amplitude'}
characteristic_unit_dict = {'DAP_amp': 'mV', 'DAP_deflection': 'mV', 'DAP_width': 'ms',
                            'DAP_time': 'ms', 'fAHP_amp': 'mV'}


def vertical_square_bracket(ax, star, x1, x2, y1, y2):
    ax.plot([x1, x2, x2, x2 + 0.1, x2, x2, x1], [y1, y1, (y1 + y2) * 0.5, (y1 + y2) * 0.5, (y1 + y2) * 0.5, y2, y2],
            lw=1.5, c='k')
    ax.text(x2 + 0.2, (y1 + y2) * 0.5, star, va='center', color='k', fontsize=14)


def get_star_from_ttest(data, h0=0):
    t, p = ttest_1samp(data[~np.isnan(data)], h0)
    star_idx = np.where([p < 0.01, p < 0.001, p < 0.0001])[0]
    if len(star_idx) == 0:
        star_idx = 0
    else:
        star_idx = star_idx[-1] + 1
    stars = ['', '*', '**', '***']
    star = stars[star_idx]
    return star


def plot_linear_fit_results(x_per_cell, xlabel, spike_characteristic_mat_per_cell, return_characteristics,
                            cell_ids, save_dir):
    slopes = init_nan((len(spike_characteristic_mat_per_cell), len(return_characteristics)))
    intercepts = init_nan((len(spike_characteristic_mat_per_cell), len(return_characteristics)))
    rmses = init_nan((len(spike_characteristic_mat_per_cell), len(return_characteristics)))
    for c_idx, characteristic in enumerate(return_characteristics):
        spike_characteristic_per_cell = [s_c[:, c_idx] for s_c in spike_characteristic_mat_per_cell]
        slopes[:, c_idx], intercepts[:, c_idx], rmses[:, c_idx] = linear_fit(spike_characteristic_per_cell, x_per_cell)
        plot_linear_fits(x_per_cell, xlabel, spike_characteristic_per_cell, return_characteristics[c_idx], cell_ids,
                         intercepts[:, c_idx], slopes[:, c_idx], save_dir)
    return slopes, intercepts, rmses


def linear_fit(spike_characteristic_per_cell, amps_per_cell):
    slopes = np.zeros(len(spike_characteristic_per_cell))
    intercepts = np.zeros(len(spike_characteristic_per_cell))
    rmse = np.zeros(len(spike_characteristic_per_cell))
    for cell_idx, (spike_characteristic, amps) in enumerate(zip(spike_characteristic_per_cell, amps_per_cell)):
        not_nan = ~np.isnan(spike_characteristic)
        if np.sum(not_nan) <= 1:  # cannot fit line with less than 2 points
            slopes[cell_idx] = intercepts[cell_idx] = rmse[cell_idx] = np.nan
        else:
            slopes[cell_idx], intercepts[cell_idx], _, _, _ = linregress(amps[not_nan], spike_characteristic[not_nan])
            rmse[cell_idx] = np.sqrt(np.mean(
                ((amps[not_nan] * slopes[cell_idx] + intercepts[cell_idx]) - spike_characteristic[not_nan]) ** 2))
    return slopes, intercepts, rmse


def plot_linear_fits(x_per_cell, xlabel, spike_characteristic_per_cell, return_characteristic, cell_ids, intercepts,
                     slopes, save_dir):

    pl.figure()
    cmap = pl.cm.get_cmap('jet')
    colors = [cmap(x) for x in np.linspace(0, 1, len(spike_characteristic_per_cell))]
    for cell_idx, (spike_characteristic, x) in enumerate(zip(spike_characteristic_per_cell, x_per_cell)):
        if ~np.isnan(slopes[cell_idx]):
            not_nan = ~np.isnan(spike_characteristic)
            pl.plot(x[not_nan], spike_characteristic[not_nan], color=colors[cell_idx], marker='o', markersize=8,
                    label=cell_ids[cell_idx])
            pl.plot(x[not_nan], x[not_nan] * slopes[cell_idx] + intercepts[cell_idx], color=colors[cell_idx],
                    linestyle=(0, (1, 1)))
    pl.ylabel(characteristic_name_dict[return_characteristic]+' ('+characteristic_unit_dict[return_characteristic]+')')
    pl.xlabel(xlabel)
    #pl.legend(fontsize=10)
    pl.tight_layout()
    if save_dir is not None:
        save_dir_img = os.path.join(save_dir, xlabel)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)
        pl.savefig(os.path.join(save_dir_img, 'linear_fits_' + return_characteristic + '.png'))
    #pl.show()


def plot_slope_rmse(return_characteristics, xlabel, rmses, slopes, slopes_model=None, rmses_model=None, model_ids=None,
                    save_dir=None):

    for c_idx, return_characteristic in enumerate(return_characteristics):
        star = get_star_from_ttest(slopes[:, c_idx])

        print 'Mean Slope %s: %.2f' % (characteristic_name_dict[return_characteristic], np.nanmean(slopes[:, c_idx]))
        fig, ax = pl.subplots(1, 2, figsize=(4.4, 4.8))
        ax[0].errorbar(0.25, np.nanmean(slopes[:, c_idx]), yerr=np.nanstd(slopes[:, c_idx]), color='k', marker='o', capsize=3)
        ax[0].plot(np.zeros(len(slopes[:, c_idx])), slopes[:, c_idx], 'ok', alpha=0.5)
        if not star == '':
            vertical_square_bracket(ax[0], star, x1=0.4, x2=0.45, y1=np.nanmean(slopes[:, c_idx]), y2=0)

        if slopes_model is not None:
            ax[0].plot(np.zeros(len(slopes_model[:, c_idx])), slopes_model[:, c_idx], 'or', alpha=0.5)
            for model_idx, model_id in enumerate(model_ids):
                ax[0].annotate(str(model_id), xy=(0.03, slopes_model[model_idx, c_idx] + 0.1), color='r', fontsize=8)

        ax[0].set_xticks([], [])
        ax[0].set_xlim([-1, 1])
        ax[0].set_ylabel('$Slope_{%s}$ (%s/nA)' % (characteristic_name_dict[return_characteristic].replace(' ', '\ '),
                                                   characteristic_unit_dict[return_characteristic]))
        ax[1].errorbar(0.25, np.nanmean(rmses[:, c_idx]), yerr=np.nanstd(rmses[:, c_idx]), color='k', marker='o', capsize=3)
        ax[1].plot(np.zeros(len(rmses[:, c_idx])), rmses[:, c_idx], 'ok', alpha=0.5)

        if rmses_model is not None:
            ax[1].plot(np.zeros(len(rmses_model[:, c_idx])), rmses_model[:, c_idx], 'or')
            for model_idx, model_id in enumerate(model_ids):
                ax[1].annotate(str(model_id), xy=(0.003, rmses_model[model_idx, c_idx] + 0.01), color='r', fontsize=8)

        ax[1].set_xticks([])
        ax[1].set_ylabel('$RMSE_{%s}$ (%s)' % (characteristic_name_dict[return_characteristic].replace(' ', '\ '),
                                                   characteristic_unit_dict[return_characteristic]))
        ax[1].set_xlim([-1, 1])
        ax[1].set_ylim([0, 2])
        pl.tight_layout()
        if save_dir is not None:
            save_dir_img = os.path.join(save_dir, xlabel)
            if not os.path.exists(save_dir_img):
                os.makedirs(save_dir_img)
            pl.savefig(os.path.join(save_dir_img, 'slope_rmse_' + return_characteristic + '.png'))
        #pl.show()


if __name__ == '__main__':
    # parameters
    save_dir = '../plots/hyper_depo/summary'
    save_dir_img = os.path.join(save_dir, 'img')

    # load
    spike_characteristic_mat_per_cell = np.load(os.path.join(save_dir, 'spike_characteristic_mat_per_cell.npy'))
    amps_per_cell = np.load(os.path.join(save_dir, 'amps_per_cell.npy'))
    return_characteristics = np.load(os.path.join(save_dir, 'return_characteristics.npy'))
    cell_ids = np.load(os.path.join(save_dir, 'cell_ids.npy'))
    v_step_per_cell = np.load(os.path.join(save_dir, 'v_step_per_cell.npy'))

    # plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    slopes, intercepts, rmses = plot_linear_fit_results(amps_per_cell, 'Step Current Amplitude (nA)', spike_characteristic_mat_per_cell,
                            return_characteristics, cell_ids, save_dir_img)

    plot_slope_rmse(return_characteristics, 'Step Current Amplitude (nA)', rmses, slopes, save_dir_img)

    slopes, intercepts, rmses = plot_linear_fit_results(v_step_per_cell, 'Mean Mem. Pot. During Step (mV)', spike_characteristic_mat_per_cell,
                            return_characteristics, cell_ids,
                            save_dir_img)
    plot_slope_rmse(return_characteristics, 'Mean Mem. Pot. During Step (mV)', rmses, slopes, save_dir_img)