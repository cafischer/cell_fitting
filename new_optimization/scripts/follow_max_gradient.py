from new_optimization.fitter.hodgkinhuxleyfitter import *
from new_optimization.optimizer.scipy_optimizer import *
from new_optimization.optimizer import OptimizerFactory
import os
from time import time

save_dir = '../../results/fitnesslandscapes/follow_max_gradient/APamp+vrest+vtrace_withweighting100_withgpas/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 5
stop_criterion = ['generation_termination', 200]
seed = 1.0  #time()
generator = 'get_random_numbers_in_bounds'
bounds = {'lower_bounds': [0, 0, 0], 'upper_bounds': [0.5, 0.4, 0.1]}
variable_keys = [[['soma', '0.5', 'na_hh', 'gnabar']],
                 [['soma', '0.5', 'k_hh', 'gkbar']],
                 [['soma', '0.5', 'pas', 'g']]
                 ]
errfun = 'rms'
fitfun = ['get_APamp', 'get_vrest', 'get_v']
fitnessweights = [1, 1, 1]
model_dir = '../../model/cells/hhCell.json'
mechanism_dir = '../../model/channels/hodgkinhuxley'
data_dir = '../../data/toymodels/hhCell/ramp.csv'
args = {'threshold': -30}

fitter = HodgkinHuxleyFitterWithFitfunList(variable_keys, errfun, fitfun, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params={'celsius': 6.3}, args=args)
#fitter = HodgkinHuxleyFitter(variable_keys, errfun, fitfun, fitnessweights,
#                 model_dir, mechanism_dir, data_dir, simulation_params={'celsius': 6.3}, args=args)

algorithm_name = 'L-BFGS-B'
algorithm_params = {'step': 1e-12}
normalize = False

optimization_params = None
save_dir_results = save_dir + '/'+algorithm_name+'/'
if not os.path.exists(save_dir_results):
    os.makedirs(save_dir_results)

optimization_settings = OptimizationSettings(maximize, n_candidates, stop_criterion, seed, generator, bounds, fitter)
algorithm_settings = AlgorithmSettings(algorithm_name, algorithm_params, optimization_params, normalize, save_dir_results)

optimizer = ScipyMaxOptimizer(optimization_settings, algorithm_settings)
#optimizer = OptimizerFactory().make_optimizer(optimization_settings, algorithm_settings)
optimizer.save(save_dir_results)
optimizer.optimize()