from new_optimization.optimizer import OptimizerFactory
from new_optimization import AlgorithmSettings, OptimizationSettings
from optimization.helpers import get_lowerbound_upperbound_keys
import os
from time import time

save_dir = '../../../results/ion_channels/nap_cubic/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 5000
stop_criterion = ['generation_termination', 1000]
seed = time()
generator = 'get_random_numbers_in_bounds'


variables = [
            [-50, -30, 'vh_m'],
            [-20, -0.1, 'k_m'],
            [-1, -0.000001, 'a_alpha_h'],
            [-100, 100, 'b_alpha_h'],
            [0.01, 50, 'k_alpha_h'],
            [0.000001, 1, 'a_beta_h'],
            [-100, 100, 'b_beta_h'],
            [-50, 0.01, 'k_beta_h'],
            ]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}


fitter_params = {
                    'name': 'ChannelFitterAllTraces',
                    'data_dir': 'plots/digitized_vsteps/traces_interpolate_cubic.csv',
                    'variable_names': variable_keys,
                    'fixed_params': {'p': 1, 'q': 1, 'h0': 1, 'e_ion': 63},
                    'n_params': len(variable_keys),
                    'compute_current_name': 'compute_current_instantaneous_m_explicit_tau'
                }

algorithm_name = 'L-BFGS-B'
algorithm_params = {}
normalize = False
optimization_params = None

save_dir = save_dir + '/' + algorithm_name + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

optimization_settings = OptimizationSettings(maximize, n_candidates, stop_criterion,
                                             seed, generator, bounds, fitter_params)
algorithm_settings = AlgorithmSettings(algorithm_name, algorithm_params, optimization_params, normalize, save_dir)

optimizer = OptimizerFactory().make_optimizer(optimization_settings, algorithm_settings)
optimizer.save(save_dir)
optimizer.optimize()