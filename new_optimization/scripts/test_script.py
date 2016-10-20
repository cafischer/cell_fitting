from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter
from new_optimization.optimizers.inspyred_optimizer import *
from new_optimization.optimizers.scipy_optimizer import *
from new_optimization.optimizers.random_optimizer import *
import os

save_dir = '../../results/test_new_opt/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 5
stop_criterion = ['generation_termination', 3]
seed = 1.1
generator = 'get_random_numbers_in_bounds'
bounds = {'lower_bounds': [0, 0], 'upper_bounds': [2, 2]}

variable_keys = [[['soma', '0.5', 'na_hh', 'gnabar']],
                 [['soma', '0.5', 'k_hh', 'gkbar']]]
errfun = 'rms'
fitfun = 'get_v'
fitnessweights = [1]
model_dir = '../../model/cells/hhCell.json'
mechanism_dir = '../../model/channels/hodgkinhuxley'
data_dir = '../../data/toymodels/hhCell/ramp.csv'

fitter = HodgkinHuxleyFitter(variable_keys, errfun, fitfun, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params={'celsius': 6.3})

#algorithm_name = 'DEA'
#algorithm_params = {'num_selected': 335, 'tournament_size': 180, 'crossover_rate': 0.57,
#                    'mutation_rate': 0.52, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.21}
#algorithm_name = 'SA'
#algorithm_params = {'temperature': 524.25, 'cooling_rate': 0.51, 'mutation_rate': 0.68,
#                    'gaussian_mean': 0.0, 'gaussian_stdev': 0.20}
#algorithm_name = 'PSO'
#algorithm_params = {'inertia': 0.43, 'cognitive_rate': 1.44, 'social_rate': 1.57}

#algorithm_name = 'L-BFGS-B'
#algorithm_params = {}

algorithm_name = 'Nelder-Mead'
algorithm_params = {}
normalize = False

#algorithm_name = 'Random'
#algorithm_params = {}
#normalize = False

optimization_params = None
save_dir_results = save_dir + '/'+algorithm_name+'/'
if not os.path.exists(save_dir_results):
    os.makedirs(save_dir_results)

optimization_settings = OptimizationSettings(maximize, n_candidates, stop_criterion, seed, generator, bounds, fitter)
with open(save_dir+'/optimization_settings.json', 'w') as f:
    optimization_settings.save(f)

algorithm_settings = AlgorithmSettings(algorithm_name, algorithm_params, optimization_params, normalize, save_dir_results)
with open(save_dir+'/algorithm_settings.json', 'w') as f:
    optimization_settings.save(f)

#optimizer = InspyredOptimizer(optimization_settings, algorithm_settings)
#optimizer = SimulatedAnnealing(optimization_settings, algorithm_settings)
optimizer = ScipyOptimizer(optimization_settings, algorithm_settings)
#optimizer = RandomOptimizer(optimization_settings, algorithm_settings)
optimizer.optimize()