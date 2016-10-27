from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter
from new_optimization.optimizers.scipy_optimizer import *
import os
from time import time
import numpy as np
from optimization.fitfuns import get_APtime

save_dir = '../../results/fitness_landscape/find_local_minima/gna_gk/rms/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 100
stop_criterion = ['generation_termination', 200]
seed = time()
generator = 'get_random_numbers_in_bounds'
bounds = {'lower_bounds': [0, 0], 'upper_bounds': [1, 1]}
variable_keys = [[['soma', '0.5', 'na_hh', 'gnabar']],
                 [['soma', '0.5', 'k_hh', 'gkbar']]]
errfun = 'rms'
fitfun = 'get_v'
#fitfun = 'shift_AP_max_APdata'
#fitfun = 'get_APamp'
fitnessweights = [1]
model_dir = '../../model/cells/hhCell.json'
mechanism_dir = '../../model/channels/hodgkinhuxley'
data_dir = '../../data/toymodels/hhCell/ramp.csv'

window_before = 5
window_after = 20
threshold = -30
data = pd.read_csv(data_dir)
args_data = {'shift': 0, 'window_before': window_before, 'window_after': window_after, 'threshold': threshold}
AP_time_data = get_APtime(np.array(data.v), np.array(data.t), np.array(data.i), args_data)

args = {'shift': 4, 'window_before': window_before, 'window_after': window_after, 'APtime': AP_time_data[0],
        'threshold': threshold}

fitter = HodgkinHuxleyFitter(variable_keys, errfun, fitfun, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params={'celsius': 6.3}, args=args)

algorithm_name = 'CG'
algorithm_params = {'step': 1e-8}
normalize = False

optimization_params = None
save_dir_results = save_dir + '/'+algorithm_name+'/'
if not os.path.exists(save_dir_results):
    os.makedirs(save_dir_results)

optimization_settings = OptimizationSettings(maximize, n_candidates, stop_criterion, seed, generator, bounds, fitter)
with open(save_dir_results+'/optimization_settings.json', 'w') as f:
    optimization_settings.save(f)

algorithm_settings = AlgorithmSettings(algorithm_name, algorithm_params, optimization_params, normalize, save_dir_results)
with open(save_dir_results+'/algorithm_settings.json', 'w') as f:
    algorithm_settings.save(f)

optimizer = ScipyOptimizer(optimization_settings, algorithm_settings)
optimizer.optimize()