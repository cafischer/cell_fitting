from new_optimization import *
from new_optimization.optimizer import OptimizerFactory
import os

save_dir = '../../results/test/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 1
stop_criterion = ['generation_termination', 2]
seed = 1.1
generator = 'get_random_numbers_in_bounds'
bounds = {'lower_bounds': [0, 0], 'upper_bounds': [2, 2]}

variable_keys = [[['soma', '0.5', 'na_hh', 'gnabar']],
                 [['soma', '0.5', 'k_hh', 'gkbar']]]
fitter_params = {
                    'name': 'HodgkinHuxleyFitter',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'fitfun_names': ['get_v'],
                    'fitnessweights': [1],
                    'model_dir': '../../model/cells/hhCell.json',
                    'mechanism_dir': '../../model/channels/hodgkinhuxley',
                    'data_dir': '../../data/toymodels/hhCell/ramp.csv',
                    'simulation_params': {'celsius': 6.3},
                    'args': {}
                }

algorithm_name = 'DEA'
algorithm_params = {'num_selected': 335, 'tournament_size': 180, 'crossover_rate': 0.57,
                    'mutation_rate': 0.52, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.21}
normalize = False

#algorithm_name = 'SA'
#algorithm_params = {'temperature': 524.25, 'cooling_rate': 0.51, 'mutation_rate': 0.68,
#                    'gaussian_mean': 0.0, 'gaussian_stdev': 0.20}
#algorithm_name = 'PSO'
#algorithm_params = {'inertia': 0.43, 'cognitive_rate': 1.44, 'social_rate': 1.57}

#algorithm_name = 'L-BFGS-B'
#algorithm_params = {}

#algorithm_name = 'Nelder-Mead'
#algorithm_params = {}
#normalize = False

#algorithm_name = 'Random'
#algorithm_params = {}
#normalize = False

optimization_params = None
save_dir = save_dir + '/' + algorithm_name + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

optimization_settings = OptimizationSettings(maximize, n_candidates, stop_criterion, seed, generator, bounds,
                                             fitter_params)
algorithm_settings = AlgorithmSettings(algorithm_name, algorithm_params, optimization_params, normalize, save_dir)

optimizer = OptimizerFactory().make_optimizer(optimization_settings, algorithm_settings)
optimizer.save(save_dir)
optimizer.optimize()