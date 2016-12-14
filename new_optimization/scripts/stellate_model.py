from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter
from new_optimization.optimizers.inspyred_optimizer import *
from new_optimization.optimizers.scipy_optimizer import *
import os
from time import time
import numpy as np
import pandas as pd
from optimization.fitfuns import *

save_dir = '../../results/new_optimization/test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 250
stop_criterion = ['generation_termination', 100]
seed = time()
generator = 'get_random_numbers_in_bounds'

variable_keys = [
                    [['soma', '0.5', 'pas', 'g']],
                    [['soma', '0.5', 'km', 'gbar']],
                    [['soma', '0.5', 'ih_fast', 'gbar']],
                    [['soma', '0.5', 'ih_slow', 'gbar']],
                    [['soma', '0.5', 'nap', 'gbar']],
                    [['soma', '0.5', 'kdr', 'gbar']],
                    [['soma', '0.5', 'kap', 'gbar']],
                    [['soma', '0.5', 'na8st', 'gbar']]
                 ]
bounds = {'lower_bounds': [0] * len(variable_keys), 'upper_bounds': [1.5] * len(variable_keys)}

model_dir = '../../model/cells/dapmodel0.json'
mechanism_dir = '../../model/channels/schmidthieber'
data_dir = '../../data/2015_08_26b/rampIV/3.0(nA).csv'
data = pd.read_csv(data_dir)
errfun = 'rms'
#fitfun = 'get_v'
fitfun = 'shifted_AP'
fitnessweights = [1]
APtime = get_APtime(np.array(data.v), np.array(data.t), np.array(data.i), None)
args = {'APtime': APtime[0], 'shift': 5, 'window_before': 2, 'window_after': 20}
#args = None
fitter = HodgkinHuxleyFitter(variable_keys, errfun, fitfun, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params={'celsius': 35}, args=args)

#algorithm_name = 'DEA'
#algorithm_params = {'num_selected': 335, 'tournament_size': 180, 'crossover_rate': 0.57,
#                    'mutation_rate': 0.52, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.21}
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

#algorithm_name = 'Nelder-Mead'
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
optimizer.optimize()