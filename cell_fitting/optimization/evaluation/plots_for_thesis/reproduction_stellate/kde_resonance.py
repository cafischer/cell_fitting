import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell
import json
from cell_fitting.read_heka import load_data, get_i_inj_standard_params
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.optimization.evaluation.plot_zap import simulate_and_compute_zap_characteristics
from cell_fitting.optimization.evaluation.plots_for_thesis.reproduction_stellate.kernel_density_estimate import compute_kde_and_alpha_hdr, plot_samples_and_kde
pl.style.use('paper_subplots')

if __name__ == '__main__':
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    model = '2'
    exp_cell = '2015_08_26b'
    step_amp = -0.1

    # load data
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'), mechanism_dir)
    standard_sim_params = get_standard_simulation_params()
    zap_params = get_i_inj_standard_params('Zap20')
    zap_params['tstop'] = 34000 - standard_sim_params['dt']
    zap_params['dt'] = standard_sim_params['dt']
    zap_params['offset_dur'] = zap_params['onset_dur'] - standard_sim_params['dt']
    _, _, _, _, _, res_freq_model, q_value_model = simulate_and_compute_zap_characteristics(cell, zap_params)

    res_freqs_data = np.load(os.path.join(save_dir_data_plots, 'Zap20/rat/summary', 'res_freqs.npy'))
    q_values_data = np.load(os.path.join(save_dir_data_plots, 'Zap20/rat/summary', 'q_values.npy'))

    # kde
    kernel_res, alpha_hdr_res = compute_kde_and_alpha_hdr(res_freqs_data, res_freq_model)
    kernel_q, alpha_hdr_q = compute_kde_and_alpha_hdr(q_values_data, q_value_model)

    # plots
    print 'alpha hdr (res. freq.): %.2f' % alpha_hdr_res
    print 'alpha hdr (Q-value): %.2f' % alpha_hdr_q
    plot_samples_and_kde(res_freqs_data, res_freq_model, kernel_res)
    pl.xlim(0, None)

    plot_samples_and_kde(q_values_data, q_value_model, kernel_q)
    pl.xlim(0, None)
    pl.show()