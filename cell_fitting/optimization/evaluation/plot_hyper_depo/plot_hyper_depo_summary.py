import matplotlib.pyplot as pl
from scipy.stats import linregress, ttest_1samp
import numpy as np
import os
pl.style.use('paper')


def vertical_square_bracket(ax, star, x1, x2, y1, y2):
    ax.plot([x1, x2, x2, x2 + 0.1, x2, x2, x1], [y1, y1, (y1 + y2) * 0.5, (y1 + y2) * 0.5, (y1 + y2) * 0.5, y2, y2],
            lw=1.5, c='k')
    ax.text(x2 + 0.2, (y1 + y2) * 0.5, star, va='center', color='k', fontsize=14)


def plot_hyper_depo_results_all_in_one(DAP_amps_data, DAP_amps_model, DAP_deflections_data, 
                                       DAP_deflections_model, DAP_widths_data, DAP_widths_model, save_dir):
    slopes_DAP_amp_data, _, rmse_DAP_amp_data = linear_fit(DAP_amps_data)
    slopes_DAP_deflection_data, _, rmse_DAP_deflection_data = linear_fit(DAP_deflections_data)
    slopes_DAP_width_data, _, rmse_DAP_width_data = linear_fit(DAP_widths_data)
    slopes_DAP_amp_model, _, rmse_DAP_amp_model = linear_fit(DAP_amps_model)
    slopes_DAP_deflection_model, _, rmse_DAP_deflection_model = linear_fit(DAP_deflections_model)
    slopes_DAP_width_model, _, rmse_DAP_width_model = linear_fit(DAP_widths_model)

    star_DAP_amp = get_star_from_ttest(slopes_DAP_amp_data)
    star_DAP_deflection = get_star_from_ttest(slopes_DAP_deflection_data)
    star_DAP_width = get_star_from_ttest(slopes_DAP_width_data)
    plot_slope_rmse_all_in_one(slopes_DAP_amp_data, slopes_DAP_deflection_data, slopes_DAP_width_data,
                               slopes_DAP_amp_model, slopes_DAP_deflection_model, slopes_DAP_width_model,
                               star_DAP_amp, star_DAP_deflection, star_DAP_width, 'Slope', save_dir)
    plot_slope_rmse_all_in_one(rmse_DAP_amp_data, rmse_DAP_deflection_data, rmse_DAP_width_data,
                               rmse_DAP_amp_model, rmse_DAP_deflection_model, rmse_DAP_width_model,
                               star_DAP_amp, star_DAP_deflection, star_DAP_width, 'RMSE', save_dir)


def plot_hyper_depo_results(DAP_characteristic_data, DAP_characteristic_model, characteristic_name, unit, save_dir):
    slopes_data, intercepts_data, rmse_data = linear_fit(DAP_characteristic_data)
    slopes_model, intercepts_model, rmse_model = linear_fit(DAP_characteristic_model)

    plot_linear_fits(DAP_characteristic_data, intercepts_data, slopes_data, characteristic_name, unit, os.path.join(save_dir, 'data_'))
    plot_linear_fits(DAP_characteristic_model, intercepts_model, slopes_model, characteristic_name, unit, os.path.join(save_dir, 'model_'))

    star = get_star_from_ttest(slopes_data)
    plot_slope_rmse(characteristic_name, rmse_data, rmse_model, slopes_data, slopes_model, star, unit, save_dir)


def get_star_from_ttest(data, h0=0):
    t, p = ttest_1samp(data, h0)
    star_idx = np.where([p < 0.01, p < 0.001, p < 0.0001])[0]
    if len(star_idx) == 0:
        star_idx = 0
    else:
        star_idx = star_idx[-1] + 1
    stars = ['', '*', '**', '***']
    star = stars[star_idx]
    return star


def plot_linear_fits(DAP_characteristic, intercepts, slopes, characteristic_name, unit, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pl.figure()
    cmap = pl.cm.get_cmap('jet')
    colors = [cmap(x) for x in np.linspace(0, 1, len(DAP_characteristic))]
    for i, DAP_characteristic_cell in enumerate(DAP_characteristic):
        pl.plot(amps, DAP_characteristic_cell, color=colors[i], marker='o', markersize=10, label=cells[i])
        pl.plot(amps, amps * slopes[i] + intercepts[i], color=colors[i])
    pl.ylabel(characteristic_name+' ('+unit+')')
    pl.xlabel('Step Current Amplitude (nA)')
    pl.tight_layout()
    pl.savefig(save_dir+ characteristic_name+'_linear_fits.png')
    pl.show()


def linear_fit(DAP_characteristic):
    slopes = np.zeros(len(DAP_characteristic))
    intercepts = np.zeros(len(DAP_characteristic))
    rmse = np.zeros(len(DAP_characteristic))
    for i, DAP_characteristic_cell in enumerate(DAP_characteristic):
        not_nan = ~np.isnan(DAP_characteristic_cell)
        if np.sum(not_nan) <= 1:
            slopes[i] = intercepts[i] = rmse[i] = np.nan
        else:
            slopes[i], intercepts[i], _, _, _ = linregress(amps[not_nan], DAP_characteristic_cell[not_nan])
            rmse[i] = np.sqrt(np.mean(
                ((amps[not_nan] * slopes[i] + intercepts[i]) - DAP_characteristic_cell[not_nan]) ** 2))
    return slopes, intercepts, rmse


def plot_slope_rmse(characteristic_name, rmse_data, rmse_model, slopes_data, slopes_model, star, unit, save_dir):
    fig, ax = pl.subplots(1, 2)
    ax[0].errorbar(0.2, np.mean(slopes_data), yerr=np.std(slopes_data), color='k', marker='o', capsize=3)
    ax[0].plot(np.zeros(len(slopes_data)), slopes_data, 'ok', alpha=0.5)
    ax[0].plot(np.zeros(len(slopes_model)), slopes_model, 'or', alpha=0.5)
    for i, model_id in enumerate(model_ids):
        ax[0].annotate(str(model_id), xy=(0.03, slopes_model[i] + 0.1), color='r', fontsize=8)
    ax[0].errorbar(1, 0, yerr=0, color='k', marker='o')
    ax[0].annotate(star, xy=(0.5, 0.75), arrowprops=dict(arrowstyle='-[, widthB=4.8, lengthB=0.4', lw=1.5),
                   ha='center', fontsize=14)
    ax[0].set_xticks([], [])
    ax[0].set_ylabel('Slope (' + unit + '/nA)')
    ax[1].errorbar(0.2, np.mean(rmse_data), yerr=np.std(rmse_data), color='k', marker='o', capsize=3)
    ax[1].plot(np.zeros(len(rmse_data)), rmse_data, 'ok', alpha=0.5)
    ax[1].plot(np.zeros(len(rmse_model)), rmse_model, 'or')
    for i, model_id in enumerate(model_ids):
        ax[1].annotate(str(model_id), xy=(0.003, rmse_model[i] + 0.01), color='r', fontsize=8)
    ax[1].set_xticks([])
    ax[1].set_ylabel('RMSE (' + unit + ')')
    ax[1].set_ylim([0, 2])
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, characteristic_name + '_slope_statistics.png'))
    pl.show()


def plot_slope_rmse_all_in_one(slopes_DAP_amp_data, slopes_DAP_deflection_data, slopes_DAP_width_data,
                               slopes_DAP_amp_model, slopes_DAP_deflection_model, slopes_DAP_width_model,
                               star_DAP_amp, star_DAP_deflection, star_DAP_width, name_statistic, save_dir):
    if name_statistic == 'Slope':
        shift_annotation = 0.25
    elif name_statistic == 'RMSE':
        shift_annotation = 0.005
    else:
        raise ValueError('Unkown statistic')

    fig, ax = pl.subplots(1, 3)
    ax[0].errorbar(0.25, np.mean(slopes_DAP_amp_data), yerr=np.std(slopes_DAP_amp_data), color='k', marker='o', capsize=3)
    ax[1].errorbar(0.25, np.mean(slopes_DAP_deflection_data), yerr=np.std(slopes_DAP_deflection_data), color='k', marker='o', capsize=3)
    ax[2].errorbar(0.25, np.mean(slopes_DAP_width_data), yerr=np.std(slopes_DAP_width_data), color='k', marker='o', capsize=3)

    ax[0].plot(np.zeros(len(slopes_DAP_amp_data)), slopes_DAP_amp_data, 'ok', alpha=0.5)
    ax[1].plot(np.zeros(len(slopes_DAP_deflection_data)), slopes_DAP_deflection_data, 'ok', alpha=0.5)
    ax[2].plot(np.zeros(len(slopes_DAP_width_data)), slopes_DAP_width_data, 'ok', alpha=0.5)

    ax[0].plot(np.zeros(len(slopes_DAP_amp_model)), slopes_DAP_amp_model, 'or', alpha=0.5)
    ax[1].plot(np.zeros(len(slopes_DAP_deflection_model)), slopes_DAP_deflection_model, 'or', alpha=0.5)
    ax[2].plot(np.zeros(len(slopes_DAP_width_model)), slopes_DAP_width_model, 'or', alpha=0.5)
    for i, model_id in enumerate(model_ids):
        ax[0].annotate(str(model_id), xy=(0.05, slopes_DAP_amp_model[i] + shift_annotation), color='r', fontsize=8)
        ax[1].annotate(str(model_id), xy=(0.05, slopes_DAP_deflection_model[i] + shift_annotation), color='r', fontsize=8)
        ax[2].annotate(str(model_id), xy=(0.05, slopes_DAP_width_model[i] + shift_annotation + 1.0*shift_annotation), color='r', fontsize=8)

    vertical_square_bracket(ax[0], star_DAP_amp, x1=0.4, x2=0.45, y1=np.mean(slopes_DAP_amp_data), y2=0)
    vertical_square_bracket(ax[1], star_DAP_deflection, x1=0.4, x2=0.45, y1=np.mean(slopes_DAP_deflection_data), y2=0)
    vertical_square_bracket(ax[2], star_DAP_width, x1=0.4, x2=0.45, y1=np.mean(slopes_DAP_width_data), y2=0)

    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    ax[0].set_ylabel(name_statistic+'$_{DAP\ Amplitude}$ (mV/nA)')
    ax[1].set_ylabel(name_statistic+'$_{DAP\ Deflection}$ (mV/nA)')
    ax[2].set_ylabel(name_statistic+'$_{DAP\ Width}$ (ms/nA)')
    ax[0].set_xlim([-1, 1])
    ax[1].set_xlim([-1, 1])
    ax[2].set_xlim([-1, 1])
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, name_statistic+'_statistics.png'))
    pl.show()


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/img/hyper_depo'
    save_dir_summary_data = '../../../data/plots/hyper_depo/summary'
    model_ids = range(1, 7)

    # load
    DAP_amps_data = np.load(os.path.join(save_dir_summary_data, 'DAP_amps.npy'))
    DAP_deflections_data = np.load(os.path.join(save_dir_summary_data, 'DAP_deflections.npy'))
    DAP_widths_data = np.load(os.path.join(save_dir_summary_data, 'DAP_widths.npy'))
    amps = np.load(os.path.join(save_dir_summary_data, 'amps.npy'))
    cells = np.load(os.path.join(save_dir_summary_data, 'cells.npy'))

    DAP_amps_model = np.zeros((len(model_ids), len(amps)))
    DAP_deflections_model = np.zeros((len(model_ids), len(amps)))
    DAP_widths_model = np.zeros((len(model_ids), len(amps)))
    for i, model_id in enumerate(model_ids):
        save_dir_summary_model = os.path.join('../../../results/best_models/', str(model_id), 'img', 'hyper_depo')
        DAP_amps_model[i, :] = np.load(os.path.join(save_dir_summary_model, 'DAP_amps.npy'))
        DAP_deflections_model[i, :] = np.load(os.path.join(save_dir_summary_model, 'DAP_deflections.npy'))
        DAP_widths_model[i, :] = np.load(os.path.join(save_dir_summary_model, 'DAP_widths.npy'))

    # plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_hyper_depo_results(DAP_amps_data, DAP_amps_model, 'DAP Amplitude', 'mV', save_dir)
    plot_hyper_depo_results(DAP_deflections_data, DAP_deflections_model, 'DAP Deflection', 'mV', save_dir)
    plot_hyper_depo_results(DAP_widths_data, DAP_widths_model, 'DAP Width', 'ms', save_dir)

    plot_hyper_depo_results_all_in_one(DAP_amps_data, DAP_amps_model, DAP_deflections_data,
                                       DAP_deflections_model, DAP_widths_data, DAP_widths_model, save_dir)