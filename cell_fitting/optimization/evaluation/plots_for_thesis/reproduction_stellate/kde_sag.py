import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell
import json
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.optimization.evaluation import simulate_model
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import compute_v_sag_and_steady_state
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
    v_model, t_model, i_inj_model = simulate_model(cell, 'IV', step_amp, 1149.95, **get_standard_simulation_params())
    start_step_idx_model = np.nonzero(i_inj_model)[0][0]
    end_step_idx_model = np.nonzero(i_inj_model)[0][-1] + 1
    v_sags, v_steady_states, _ = compute_v_sag_and_steady_state([v_model], [step_amp], AP_threshold=0,
                                                                start_step_idx=start_step_idx_model,
                                                                end_step_idx=end_step_idx_model)
    vrest_model = np.mean(v_model[:start_step_idx_model])
    sag_deflection_model = v_steady_states[0] - v_sags[0]
    steady_state_amp_model = vrest_model - v_steady_states[0]

    sag_deflections_data = np.load(os.path.join(save_dir_data_plots, 'IV', 'sag', 'rat', str(step_amp),
                                                'sag_amps.npy'))
    steady_state_amps_data = np.load(os.path.join(save_dir_data_plots, 'IV', 'sag', 'rat', str(step_amp),
                                                  'v_deflections.npy'))

    # kde
    kernel_sag, alpha_hdr_sag = compute_kde_and_alpha_hdr(sag_deflections_data, sag_deflection_model)
    kernel_ss, alpha_hdr_ss = compute_kde_and_alpha_hdr(steady_state_amps_data, steady_state_amp_model)

    # plots
    print 'alpha hdr (sag defl.): %.2f' % alpha_hdr_sag
    print 'alpha hdr (steady state amp.): %.2f' % alpha_hdr_ss
    plot_samples_and_kde(sag_deflections_data, sag_deflection_model, kernel_sag)
    pl.xlim(0, None)

    plot_samples_and_kde(steady_state_amps_data, steady_state_amp_model, kernel_ss)
    pl.xlim(0, None)
    pl.show()
