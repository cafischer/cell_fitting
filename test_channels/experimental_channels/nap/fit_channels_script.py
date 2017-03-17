from new_optimization.optimizer import OptimizerFactory
from new_optimization import AlgorithmSettings, OptimizationSettings
from optimization.helpers import get_lowerbound_upperbound_keys
import os
import numpy as np
from time import time

save_dir = '../../results/new_optimization/ion_channels/nap/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 100
stop_criterion = ['generation_termination', 500]
seed = time()
generator = 'get_random_numbers_in_bounds'


variables = [
                    [0, 1, 'm_inf'],
                    [0, 1, 'h_inf'],
                    [0, 1000, 'tau_h']
            ]
v_steps = np.arange(-60, -34, 5)
lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
lower_bounds = np.ravel([lower_bounds for i in range(len(v_steps))])
upper_bounds = np.ravel([upper_bounds for i in range(len(v_steps))])
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}


fitter_params = {
                    'name': 'ChannelFitter',
                    'data_dir': 'plots/digitized_vsteps/traces.csv',
                    'fixed_params': {'p': 1, 'q': 1, 'h0': 1, 'e_ion': 63},
                    'n_params': 3,
                    'compute_current_name': 'compute_current_instantaneous_m'
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