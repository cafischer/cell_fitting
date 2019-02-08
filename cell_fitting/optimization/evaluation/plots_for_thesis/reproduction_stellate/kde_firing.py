import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell
import json
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.optimization.evaluation import simulate_model
from cell_fitting.optimization.evaluation.plot_IV.latency_vs_ISI12_distribution import get_latency_and_ISI12
from cell_fitting.optimization.evaluation.plot_IV import simulate_and_compute_fI_curve, fit_fI_curve
from cell_fitting.optimization.evaluation.plots_for_thesis.reproduction_stellate.kernel_density_estimate import compute_kde_and_alpha_hdr, plot_samples_and_kde
pl.style.use('paper_subplots')

if __name__ == '__main__':
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    model = '2'
    exp_cell = '2015_08_26b'
    step_amp = -0.1

    # latency vs ISI1/2

    # load data
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'), mechanism_dir)
    latency_model, ISI12_model = get_latency_and_ISI12(cell)

    latency_data = np.load(os.path.join(save_dir_data_plots, 'IV/latency_vs_ISI12/rat', 'latency.npy'))
    ISI12_data = np.load(os.path.join(save_dir_data_plots, 'IV/latency_vs_ISI12/rat', 'ISI12.npy'))
    latency_data = latency_data[latency_data >= 0]
    ISI12_data = ISI12_data[latency_data >= 0]

    # kde
    kernel_latency, alpha_hdr_latency = compute_kde_and_alpha_hdr(latency_data, latency_model)
    kernel_ISI12, alpha_hdr_ISI12 = compute_kde_and_alpha_hdr(ISI12_data, ISI12_model)

    # plots
    print 'alpha hdr (latency of 1st spike): %.2f' % alpha_hdr_latency
    print 'alpha hdr (ISI12): %.2f' % alpha_hdr_ISI12
    plot_samples_and_kde(latency_data, latency_model, kernel_latency)
    pl.xlim(0, None)

    plot_samples_and_kde(ISI12_data, ISI12_model, kernel_ISI12)
    pl.xlim(0, None)
    pl.show()

    # fit F-I curve

    # load data
    amps_greater0, firing_rates_model = simulate_and_compute_fI_curve(cell)
    FI_a_model, FI_b_model, FI_c_model, RMSE_model = fit_fI_curve(amps_greater0, firing_rates_model)

    FI_a = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'FI_a.npy'))
    FI_b = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'FI_b.npy'))
    FI_c = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'FI_c.npy'))
    RMSE = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'RMSE.npy'))

    # kde
    kernel_a, alpha_hdr_a= compute_kde_and_alpha_hdr(FI_a, FI_a_model)
    kernel_b, alpha_hdr_b = compute_kde_and_alpha_hdr(FI_b, FI_b_model)
    kernel_c, alpha_hdr_c = compute_kde_and_alpha_hdr(FI_c, FI_c_model)

    # plots
    print 'alpha hdr (a): %.2f' % alpha_hdr_a
    print 'alpha hdr (b): %.2f' % alpha_hdr_b
    print 'alpha hdr (c): %.2f' % alpha_hdr_c
    plot_samples_and_kde(FI_a, FI_a_model, kernel_a)
    pl.xlim(0, None)

    plot_samples_and_kde(FI_b, FI_b_model, kernel_b)
    pl.xlim(0, None)

    plot_samples_and_kde(FI_c, FI_c_model, kernel_c)
    pl.xlim(0, None)
    pl.show()