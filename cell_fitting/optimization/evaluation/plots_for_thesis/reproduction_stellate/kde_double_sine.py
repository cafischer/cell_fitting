import matplotlib.pyplot as pl
import numpy as np
import os
import json
import scipy.stats as st
from cell_fitting.optimization.evaluation.plots_for_thesis.reproduction_stellate.kernel_density_estimate import compute_kde_and_alpha_hdr, plot_samples_and_kde
pl.style.use('paper_subplots')

if __name__ == '__main__':
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    freq1 = 0.1
    freq2 = 5

    # load data
    amp1_data = None
    amp2_data = None
    phase_means_data = np.load(os.path.join(save_dir_data_plots, 'sine_stimulus', 'traces', 'rat', 'summary',
                                            'spike_phase', str(amp1_data) + '_' + str(amp2_data) + '_'
                                            + str(freq1) + '_' + str(freq2), 'phase_means.npy'))
    phase_stds_data = np.load(os.path.join(save_dir_data_plots, 'sine_stimulus', 'traces', 'rat', 'summary',
                                           'spike_phase', str(amp1_data) + '_' + str(amp2_data) +'_'
                                           + str(freq1) + '_' + str(freq2), 'phase_stds.npy'))

    # load model
    amp1 = 0.4
    amp2 = 0.4
    with open(os.path.join(save_dir_model, model, 'img', 'sine_stimulus/traces',
                           str(amp1) + '_' + str(amp2) + '_' + str(freq1) + '_' + str(freq2), 'phase_hist',
                           'sine_dict.json'), 'r') as density_est:
        sine_dict_model = json.load(density_est)
    phase_mean_model = sine_dict_model['mean_phase'][0]
    phase_std_model = sine_dict_model['std_phase'][0]

    # kde
    kernel_means, alpha_hdr_means = compute_kde_and_alpha_hdr(phase_means_data, phase_mean_model)
    kernel_stds, alpha_hdr_stds = compute_kde_and_alpha_hdr(phase_stds_data, phase_std_model)

    # plots
    print 'alpha hdr (means): %.2f' % alpha_hdr_means
    print 'alpha hdr (stds): %.2f' % alpha_hdr_stds
    plot_samples_and_kde(phase_means_data, phase_mean_model, kernel_means)
    plot_samples_and_kde(phase_stds_data, phase_std_model, kernel_stds)
    pl.show()

