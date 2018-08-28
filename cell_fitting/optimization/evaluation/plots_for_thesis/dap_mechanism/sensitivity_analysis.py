import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from nrn_wrapper import Cell
from cell_fitting.sensitivity_analysis.plot_correlation_param_characteristic import plot_corr_on_ax
pl.style.use('paper_subplots')


# TODO: colors
if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/DAP-Project/thesis/figures'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    v_init = -75
    characteristics = ['DAP_deflection', 'DAP_amp', 'DAP_width', 'DAP_time'][:2]

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'), mechanism_dir)

    # load
    correlation_measure = 'kendalltau'  # non-parametric, robust
    s_ = os.path.join('/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/sensitivity_analysis',
                      'mean_2std_6models', 'analysis', 'plots', 'correlation', 'parameter_characteristic')
    corr_mat = np.load(os.path.join(s_, 'all', 'correlation_' + correlation_measure + '.npy'))
    p_val_mat = np.load(os.path.join(s_, 'all', 'p_val_' + correlation_measure + '.npy'))
    return_characteristics = np.load(os.path.join(s_, 'all', 'return_characteristics_' + correlation_measure + '.npy'))
    parameters = np.load(os.path.join(s_, 'all', 'variable_names_' + correlation_measure + '.npy'))
    mean_mat = np.load(os.path.join(s_, 'sampled', correlation_measure, 'mean_mat.npy'))
    std_mat = np.load(os.path.join(s_, 'sampled', correlation_measure, 'std_mat.npy'))

    characteristic_idxs = [np.where(characteristic==return_characteristics)[0] for characteristic in characteristics]

    # plot
    fig = pl.figure(figsize=(12, 8))
    outer = gridspec.GridSpec(5, 1)

    # resampled mean and std of correlations
    axes = [outer[0, 0], outer[1, 0], outer[2, 0], outer[3, 0]]
    for i, (characteristic_idx, characteristic) in enumerate(zip(characteristic_idxs, characteristics)):
        ax = pl.Subplot(fig, axes[i])
        fig.add_subplot(ax)

        ax.errorbar(range(len(parameters)), mean_mat[characteristic_idx, :][0], yerr=std_mat[characteristic_idx, :][0],
                    fmt='o', color='k', capsize=3)
        ax.axhline(0.5, color='0.5', linestyle='--')
        ax.axhline(0.0, color='0.5', linestyle='--')
        ax.axhline(-0.5, color='0.5', linestyle='--')
        ax.set_xticks(range(len(parameters)))
        ax.set_xticklabels(parameters, rotation='40', ha='right')
        ax.set_ylabel(characteristic)  # TODO
        ax.set_ylim(-1, 1)
        pl.xlabel('Parameter')

    # sensitivity analysis
    ax = pl.Subplot(fig, outer[4, 0])
    fig.add_subplot(ax)

    plot_corr_on_ax(ax, corr_mat, p_val_mat, characteristics, parameters, correlation_measure)
    # corrected for multiple testing with bonferroni

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'sensitivity_analysis.png'))
    pl.show()