import matplotlib.pyplot as pl
from scipy.stats import linregress
import numpy as np
import os
from cell_fitting.data.plot_hyper_depo.plot_hyper_depo_summary import plot_linear_fit_results, \
    vertical_square_bracket, get_star_from_ttest, plot_slope_rmse
pl.style.use('paper')


def plot_hyper_depo_results_all_in_one(amps, DAP_amps_data, DAP_amps_model, DAP_deflections_data,
                                       DAP_deflections_model, DAP_widths_data, DAP_widths_model, save_dir):
    slopes_DAP_amp_data, _, rmse_DAP_amp_data = linear_fit(DAP_amps_data, amps)
    slopes_DAP_deflection_data, _, rmse_DAP_deflection_data = linear_fit(DAP_deflections_data, amps)
    slopes_DAP_width_data, _, rmse_DAP_width_data = linear_fit(DAP_widths_data, amps)
    slopes_DAP_amp_model, _, rmse_DAP_amp_model = linear_fit(DAP_amps_model, amps)
    slopes_DAP_deflection_model, _, rmse_DAP_deflection_model = linear_fit(DAP_deflections_model, amps)
    slopes_DAP_width_model, _, rmse_DAP_width_model = linear_fit(DAP_widths_model, amps)

    star_DAP_amp = get_star_from_ttest(slopes_DAP_amp_data)
    star_DAP_deflection = get_star_from_ttest(slopes_DAP_deflection_data)
    star_DAP_width = get_star_from_ttest(slopes_DAP_width_data)
    plot_slope_rmse_all_in_one(slopes_DAP_amp_data, slopes_DAP_deflection_data, slopes_DAP_width_data,
                               slopes_DAP_amp_model, slopes_DAP_deflection_model, slopes_DAP_width_model,
                               star_DAP_amp, star_DAP_deflection, star_DAP_width, 'Slope', save_dir)
    plot_slope_rmse_all_in_one(rmse_DAP_amp_data, rmse_DAP_deflection_data, rmse_DAP_width_data,
                               rmse_DAP_amp_model, rmse_DAP_deflection_model, rmse_DAP_width_model,
                               star_DAP_amp, star_DAP_deflection, star_DAP_width, 'RMSE', save_dir)


def plot_linear_fits(amps, DAP_characteristic, intercepts, slopes, characteristic_name, unit, save_dir):
    pl.figure()
    cmap = pl.cm.get_cmap('jet')
    colors = [cmap(x) for x in np.linspace(0, 1, len(DAP_characteristic))]
    for i, DAP_characteristic_cell in enumerate(DAP_characteristic):
        pl.plot(amps, DAP_characteristic_cell, color=colors[i], marker='o', markersize=10)
        pl.plot(amps, amps * slopes[i] + intercepts[i], color=colors[i])
    pl.ylabel(characteristic_name+' ('+unit+')')
    pl.xlabel('Step Current Amplitude (nA)')
    pl.tight_layout()
    pl.savefig(save_dir + characteristic_name + '_linear_fits.png')
    pl.show()


def linear_fit(DAP_characteristic, amps):
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
    save_dir_img = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/img/hyper_depo'
    save_dir_summary_data = '../../../data/plots/hyper_depo/summary'
    save_dir_models = '../../../results/best_models/'
    model_ids = range(1, 7)

    # load data
    spike_characteristic_mat_per_cell = np.load(os.path.join(save_dir_summary_data,
                                                             'spike_characteristic_mat_per_cell.npy'))
    amps_per_cell = np.load(os.path.join(save_dir_summary_data, 'amps_per_cell.npy'))
    return_characteristics = np.load(os.path.join(save_dir_summary_data, 'return_characteristics.npy'))
    cell_ids = np.load(os.path.join(save_dir_summary_data, 'cell_ids.npy'))
    v_step_per_cell = np.load(os.path.join(save_dir_summary_data, 'v_step_per_cell.npy'))

    # load model
    spike_characteristic_mat_per_model = []
    amps_per_model = []
    v_step_per_model = []
    for i, model_id in enumerate(model_ids):
        save_dir_summary_model = os.path.join(save_dir_models, str(model_id), 'img', 'hyper_depo')
        spike_characteristic_mat_per_model.append(np.load(os.path.join(save_dir_summary_model,
                                                                       'spike_characteristics_mat.npy')))
        amps_per_model.append(np.load(os.path.join(save_dir_summary_model, 'amps.npy')))
        v_step_per_model.append(np.load(os.path.join(save_dir_summary_model, 'v_step.npy')))

    # plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # vs current
    slopes_model, intercepts_model, rmses_model = plot_linear_fit_results(amps_per_model, 'Step Current Amplitude (nA)',
                                                        spike_characteristic_mat_per_model, return_characteristics,
                                                        model_ids, save_dir_img)
    slopes_data, intercepts_data, rmses_data = plot_linear_fit_results(amps_per_cell, 'Step Current Amplitude (nA)',
                                                        spike_characteristic_mat_per_cell, return_characteristics,
                                                        cell_ids, None)
    plot_slope_rmse(return_characteristics, 'Step Current Amplitude (nA)', rmses_data, slopes_data,
                    slopes_model, rmses_model, model_ids, save_dir_img)

    # vs voltage
    slopes_model, intercepts_model, rmses_model = plot_linear_fit_results(v_step_per_model, 'Mean Mem. Pot. During Step (mV)',
                                                        spike_characteristic_mat_per_model, return_characteristics,
                                                        model_ids, save_dir_img)
    slopes_data, intercepts_data, rmses_data = plot_linear_fit_results(v_step_per_cell, 'Mean Mem. Pot. During Step (mV)',
                                                        spike_characteristic_mat_per_cell, return_characteristics,
                                                        cell_ids, None)
    plot_slope_rmse(return_characteristics, 'Mean Mem. Pot. During Step (mV)', rmses_data, slopes_data,
                    slopes_model, rmses_model, model_ids, save_dir_img)



    # plot_hyper_depo_results_all_in_one(amps, DAP_amps_data, DAP_amps_model, DAP_deflections_data,
    #                                    DAP_deflections_model, DAP_widths_data, DAP_widths_model, save_dir)