from new_optimization.fitter.linearregressionfitter import LinearRegressionFitter
from new_optimization import *
from new_optimization.optimizer import OptimizerFactory
from optimization.helpers import *
import os
from time import time
import pandas as pd

save_dir = '../../results/optimization/linearregression/2015_08_26b/improve_kdr/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 2000
stop_criterion = ['generation_termination', 500]
seed = 1.0
generator = 'get_random_numbers_in_bounds'

#variables = [
#            [-50, -20, [['soma', '0.5', 'nat', 'tha']]],
#            [5, 15, [['soma', '0.5', 'nat', 'qa']]],
#            [0.001, 1.0, [['soma', '0.5', 'nat', 'Ra']]],
#            [0.001, 1.0, [['soma', '0.5', 'nat', 'Rb']]],
#            [-70, -30, [['soma', '0.5', 'nat', 'thi1']]],
#            [-90, -50, [['soma', '0.5', 'nat', 'thi2']]],
#            [0.01, 10, [['soma', '0.5', 'nat', 'qi']]],
#            [-80, -40, [['soma', '0.5', 'nat', 'thinf']]],
#            [0, 100, [['soma', '0.5', 'nat', 'qinf']]],
#            [0.0001, 0.1, [['soma', '0.5', 'nat', 'Rg']]],
#            [0.0001, 0.1, [['soma', '0.5', 'nat', 'Rd']]]
#            ]

variables = [
            [-10, 30, [['soma', '0.5', 'kdr', 'vhalfn']]],
            [0.0001, 0.1, [['soma', '0.5', 'kdr', 'a0n']]],
            [-20, 10, [['soma', '0.5', 'kdr', 'zetan']]],
            [-5, 5, [['soma', '0.5', 'kdr', 'gmn']]]
]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}

model_dir = '../../model/cells/cell_improved_nat.json'
mechanism_dir = '../../model/vclamp/stellate'
data_dir = '../../data/2015_08_26b/raw/simulate_rampIV/3.0(nA).csv'
fitter = LinearRegressionFitter(variable_keys, model_dir, mechanism_dir, data_dir, simulation_params={'celsius': 35},
                                with_cm=False)

algorithm_name = 'DEA'
algorithm_params = {'num_selected': 450, 'tournament_size': 20, 'crossover_rate': 0.5,
                    'mutation_rate': 0.5, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.2}
normalize = True

#algorithm_name = 'SA'
#algorithm_params = {'temperature': 524.25, 'cooling_rate': 0.51, 'mutation_rate': 0.68,
#                    'gaussian_mean': 0.0, 'gaussian_stdev': 0.20}
#normalize = True

#algorithm_name = 'PSO'
#algorithm_params = {'inertia': 0.43, 'cognitive_rate': 1.44, 'social_rate': 1.57}
#normalize = True

#algorithm_name = 'L-BFGS-B'
#algorithm_params = {} # {'gtol': 1e-15, 'ftol': 1e-15}
#normalize = False

optimization_params = None

save_dir = save_dir + '/' + algorithm_name + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

optimization_settings = OptimizationSettings(maximize, n_candidates, stop_criterion, seed, generator, bounds, fitter)
algorithm_settings = AlgorithmSettings(algorithm_name, algorithm_params, optimization_params, normalize, save_dir)

optimizer = OptimizerFactory().make_optimizer(optimization_settings, algorithm_settings)
optimizer.save(save_dir)
optimizer.optimize()