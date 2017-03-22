from new_optimization.optimizer import OptimizerFactory
from new_optimization import AlgorithmSettings, OptimizationSettings
from optimization.helpers import get_lowerbound_upperbound_keys
import os
from time import time

save_dir = '../../../results/ion_channels/nat2/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 5000
stop_criterion = ['generation_termination', 1000]
seed = time()
generator = 'get_random_numbers_in_bounds'


variables = [
                    [0.0001, 1, 'alpha_a_m'],
                    [-100, 100, 'alpha_b_m'],
                    [-40, 0, 'alpha_k_m'],
                    [-1, -0.001, 'beta_a_m'],
                    [-100, 100, 'beta_b_m'],
                    [0, 30, 'beta_k_m'],
                    [-1, -0.001, 'alpha_a_h'],
                    [-100, 100, 'alpha_b_h'],
                    [0, 40, 'alpha_k_h'],
                    [0.001, 2, 'beta_a_h'],
                    [-100, 100, 'beta_b_h'],
                    [-30, 0, 'beta_k_h'],
            ]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}


fitter_params = {
                    'name': 'ChannelFitterAllTraces',
                    'data_dir': 'plots/digitized_vsteps/traces.csv',
                    'fixed_params': {'p': 3, 'q': 1, 'm0': 0, 'h0': 1, 'e_ion': 63},
                    'n_params': len(variable_keys),
                    'compute_current_name': 'compute_current_explicit_tau'
                }

algorithm_name = 'L-BFGS-B'
algorithm_params = {}
normalize = False
optimization_params = None

#algorithm_name = 'DEA'
#algorithm_params = {'num_selected': 90, 'tournament_size': 40, 'crossover_rate': 0.5,
#                    'mutation_rate': 0.5, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.2}
#normalize = True
#optimization_params = None

save_dir = save_dir + '/' + algorithm_name + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

optimization_settings = OptimizationSettings(maximize, n_candidates, stop_criterion,
                                             seed, generator, bounds, fitter_params)
algorithm_settings = AlgorithmSettings(algorithm_name, algorithm_params, optimization_params, normalize, save_dir)

optimizer = OptimizerFactory().make_optimizer(optimization_settings, algorithm_settings)
optimizer.save(save_dir)
optimizer.optimize()