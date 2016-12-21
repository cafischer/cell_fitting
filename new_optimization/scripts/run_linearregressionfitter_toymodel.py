from new_optimization.fitter.linearregressionfitter import LinearRegressionFitter
from new_optimization import *
from new_optimization.optimizer import OptimizerFactory
from optimization.helpers import *
import os
from time import time
import pandas as pd

save_dir = '../../results/new_optimization/linearregression/test0/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 10
stop_criterion = ['generation_termination', 100]
seed = 1.0
generator = 'get_random_numbers_in_bounds'

variables = [
                [0, 1,  [['soma', '0.5', 'na_hh', 'alpha_m_f']]],
                [0, 60, [['soma', '0.5', 'na_hh', 'alpha_m_v']]],
                [0, 50, [['soma', '0.5', 'na_hh', 'alpha_m_k']]],
                 ]
lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}

model_dir = '../../model/cells/hhCell.json'
mechanism_dir = '../../model/channels/hodgkinhuxley'
data_dir = '../../data/toymodels/hhCell/ramp.csv'
fitter = LinearRegressionFitter(variable_keys, model_dir, mechanism_dir, data_dir, simulation_params={'celsius': 6.3})

#algorithm_name = 'DEA'
#algorithm_params = {'num_selected': 200, 'tournament_size': 20, 'crossover_rate': 0.5,
#                    'mutation_rate': 0.5, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.2}
#normalize = True

#algorithm_name = 'SA'
#algorithm_params = {'temperature': 524.25, 'cooling_rate': 0.51, 'mutation_rate': 0.68,
#                    'gaussian_mean': 0.0, 'gaussian_stdev': 0.20}
#normalize = True

#algorithm_name = 'PSO'
#algorithm_params = {'inertia': 0.43, 'cognitive_rate': 1.44, 'social_rate': 1.57}
#normalize = True

algorithm_name = 'L-BFGS-B'
algorithm_params = {}
normalize = False

optimization_params = None

save_dir = save_dir + '/' + algorithm_name + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

optimization_settings = OptimizationSettings(maximize, n_candidates, stop_criterion, seed, generator, bounds, fitter)
algorithm_settings = AlgorithmSettings(algorithm_name, algorithm_params, optimization_params, normalize, save_dir)

optimizer = OptimizerFactory().make_optimizer(optimization_settings, algorithm_settings)
optimizer.save(save_dir)
optimizer.optimize()