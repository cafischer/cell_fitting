from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter
from new_optimization.optimizers.scipy_optimizer import *
import os
from time import time
import numpy as np
from optimization.fitfuns import get_APtime

save_dir = '../../results/fitnesslandscapes/find_local_minima/combined_fitfuns/gna_gk/interpolate_v_trace+penalize_not1AP+APamp/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_dir_ranges = '../../results/fitnesslandscapes/modellandscape/gna_gk/'
save_dir_fitness = '../../results/fitnesslandscapes/modellandscape/gna_gk/fitfuns/APamp+v_trace+penalize_not1AP/error.npy'
optimum = [0.12, 0.036]

maximize = False
n_candidates = 100
stop_criterion = ['generation_termination', 200]
seed = 1.11  #time()
generator = 'get_random_numbers_in_bounds'
bounds = {'lower_bounds': [0, 0], 'upper_bounds': [1, 1]}
variable_keys = [[['soma', '0.5', 'na_hh', 'gnabar']],
                 [['soma', '0.5', 'k_hh', 'gkbar']]
                 #[['soma', '0.5', 'pas', 'g']]
                 ]
errfun = 'rms'
fitfun = ['interpolate_fitness']  #['get_v', 'penalize_not1AP', 'get_APamp']
fitnessweights = [1]
model_dir = '../../model/cells/hhCell.json'
mechanism_dir = '../../model/channels/hodgkinhuxley'
data_dir = '../../data/toymodels/hhCell/ramp.csv'

window_before = 5
window_after = 20
threshold = -30
penalty = 50
with open(save_dir_fitness, 'r') as f:
    fitness = np.load(f)
p1_range = np.loadtxt(save_dir_ranges + '/p1_range.txt')
p2_range = np.loadtxt(save_dir_ranges + '/p2_range.txt')
data = pd.read_csv(data_dir)
args = {'shift': 4, 'window_before': window_before, 'window_after': window_after,
        'threshold': threshold,
        'penalty': penalty,
        'fitness': fitness, 'p1_range': p1_range, 'p2_range': p2_range, 'candidate': optimum}
AP_time_data = get_APtime(np.array(data.v), np.array(data.t), np.array(data.i), args)
args['APtime'] = AP_time_data

fitter = HodgkinHuxleyFitter(variable_keys, errfun, fitfun, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params={'celsius': 6.3}, args=args)

algorithm_name = 'L-BFGS-B'
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