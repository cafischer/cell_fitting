from new_optimization.optimizer import OptimizerFactory
from new_optimization import AlgorithmSettings, OptimizationSettings
from optimization.helpers import get_lowerbound_upperbound_keys
import os
from time import time

save_dir = '../../../results/ion_channels/nat_new/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 5000
stop_criterion = ['generation_termination', 1000]
seed = time()
generator = 'get_random_numbers_in_bounds'


variables = [
                    [0.000001, 2, 'a_alpha_m'],
                    [-100, 100, 'b_alpha_m'],
                    [-50, 0, 'k_alpha_m'],
                    [-2, -0.000001, 'a_beta_m'],
                    [-100, 100, 'b_beta_m'],
                    [0, 50, 'k_beta_m'],
                    [-2, -0.000001, 'a_alpha_h'],
                    [-100, 100, 'b_alpha_h'],
                    [0, 50, 'k_alpha_h'],
                    [0.000001, 2, 'a_beta_h'],
                    [-100, 100, 'b_beta_h'],
                    [-50, 0, 'k_beta_h'],
            ]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}


fitter_params = {
                    'name': 'ChannelFitterAllTraces',
                    'data_dir': 'plots/digitized_vsteps/traces.csv',
                    'variable_names': variable_keys,
                    'fixed_params': {'p': 3, 'q': 1, 'm0': 0, 'h0': 1, 'e_ion': 63},
                    'n_params': len(variable_keys),
                    'compute_current_name': 'compute_current_explicit_tau'
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