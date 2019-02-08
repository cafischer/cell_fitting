import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation.plot_rampIV import simulate_rampIV
from cell_fitting.optimization.evaluation.plots_for_thesis.reproduction_stellate.kernel_density_estimate import compute_kde_and_alpha_hdr, plot_samples_and_kde
pl.style.use('paper_subplots')

if __name__ == '__main__':
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    model = '2'
    ramp_amp = 3.5

    # load data
    characteristics = ['DAP_deflection', 'DAP_amp', 'DAP_time', 'DAP_width']
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'), mechanism_dir)
    v, t, i_inj = simulate_rampIV(cell, ramp_amp, v_init=-75)
    start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
    v_rest = np.mean(v[0:start_i_inj])
    characteristics_mat_model = np.array(get_spike_characteristics(v, t, characteristics, v_rest, check=False,
                                                                   **get_spike_characteristics_dict()),
                                         dtype=float)

    characteristics_mat_exp = np.load(os.path.join(save_dir_data_plots, 'spike_characteristics/rat',
                                                   'characteristics_mat.npy')).astype(float)
    characteristics_exp = np.load(os.path.join(save_dir_data_plots, 'spike_characteristics/rat',
                                               'return_characteristics.npy'))

    for characteristic_idx, characteristic in enumerate(characteristics):
        characteristic_idx_exp = np.where(characteristic == characteristics_exp)[0][0]
        not_nan_exp = ~np.isnan(characteristics_mat_exp[:, characteristic_idx_exp])
        characteristic_exp = characteristics_mat_exp[:, characteristic_idx_exp][not_nan_exp]

        # kde
        kernel, alpha_hdr = compute_kde_and_alpha_hdr(characteristic_exp, characteristics_mat_model[characteristic_idx])

        # plots
        print 'alpha hdr ('+characteristic+'): %.2f' % alpha_hdr
        plot_samples_and_kde(characteristic_exp, characteristics_mat_model[characteristic_idx], kernel)
        pl.xlim(0, None)
    pl.show()
