import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from nrn_wrapper import Cell
from cell_fitting.sensitivity_analysis.plot_correlation_param_characteristic import plot_corr_on_ax
from cell_fitting.util import characteristics_dict_for_plotting, get_variable_names_for_plotting
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    v_init = -75
    characteristics = ['DAP_deflection', 'DAP_amp', 'DAP_width', 'DAP_time']
    correlation_measure = 'kendalltau'  #kendalltau non-parametric, robust

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'), mechanism_dir)

    # load
    s__ = os.path.join('/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/sensitivity_analysis',
                      'mean_std_1order_of_mag_model2', 'analysis')
    s_ = os.path.join('/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/sensitivity_analysis',
                      'mean_std_1order_of_mag_model2', 'analysis', 'plots', 'correlation', 'parameter_characteristic')
    corr_mat = np.load(os.path.join(s_, 'all', 'correlation_' + correlation_measure + '.npy'))
    p_val_mat = np.load(os.path.join(s_, 'all', 'p_val_' + correlation_measure + '.npy'))
    return_characteristics = np.load(os.path.join(s_, 'all', 'return_characteristics_' + correlation_measure + '.npy'))
    parameters = np.load(os.path.join(s_, 'all', 'variable_names_' + correlation_measure + '.npy'))
    mean_mat = np.load(os.path.join(s_, 'sampled', correlation_measure, 'mean_mat.npy'))
    std_mat = np.load(os.path.join(s_, 'sampled', correlation_measure, 'std_mat.npy'))
    return_characteristics_std = np.load(os.path.join(s__, 'return_characteristics.npy'))

    characteristic_idxs = np.array([np.where(characteristic == return_characteristics)[0][0]
                                    for characteristic in characteristics], dtype=int)
    characteristic_idxs_std = np.array([np.where(characteristic == return_characteristics_std)[0][0]
                                        for characteristic in characteristics], dtype=int)

    # plot
    fig = pl.figure(figsize=(12, 8))
    outer = gridspec.GridSpec(2, 1, hspace=-0.15)

    # resampled mean and std of correlations
    characteristics_dict = characteristics_dict_for_plotting()
    new_parameter_names = get_variable_names_for_plotting(parameters)
    inner = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0, 0], hspace=0.25)
    axes = [inner[0, 0], inner[1, 0], inner[2, 0], inner[3, 0]]
    for i, (characteristic_idx, characteristic) in enumerate(zip(characteristic_idxs_std, characteristics)):
        ax = pl.Subplot(fig, axes[i])
        fig.add_subplot(ax)

        ax.errorbar(range(len(parameters)), mean_mat[characteristic_idx, :], yerr=std_mat[characteristic_idx, :],
                    fmt='o', color='k', capsize=3)
        ax.axhline(0.5, color='0.5', linestyle='--')
        ax.axhline(0.0, color='0.5', linestyle='--')
        ax.axhline(-0.5, color='0.5', linestyle='--')
        ax.set_xticks(range(len(parameters)))
        ax.set_xticklabels([])
        ax.set_ylabel(characteristics_dict[characteristic])
        ax.set_ylim(-1, 1)
        ax.set_xlim(-0.5, len(parameters) - 0.5)
        if i == 0:
            ax.text(-0.16, 1.0, 'A', transform=ax.transAxes, size=18, weight='bold')
    ax.set_xticklabels(new_parameter_names, rotation='40', ha='right')
    ax.set_xlabel('Parameter')

    # sensitivity analysis
    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)

    corr_mat = corr_mat[characteristic_idxs, :]
    p_val_mat = p_val_mat[characteristic_idxs, :]
    plot_corr_on_ax(ax, corr_mat, p_val_mat, characteristics, parameters, correlation_measure, cmap='seismic')
    # corrected for multiple testing with bonferroni
    ax.text(-0.17, 1.0, 'B', transform=ax.transAxes, size=18, weight='bold')

    pl.tight_layout()
    pl.subplots_adjust(top=0.97, right=0.94, left=0.13, bottom=-0.1)
    pl.savefig(os.path.join(save_dir_img, 'sensitivity_analysis.png'))
    pl.show()