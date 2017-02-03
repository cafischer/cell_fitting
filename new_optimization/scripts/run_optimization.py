from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter, HodgkinHuxleyFitterPareto
from new_optimization import *
from new_optimization.optimizer import OptimizerFactory
import os
from time import time
import pandas as pd

save_dir = '../../results/new_optimization/2015_08_26b/test/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 1000
stop_criterion = ['generation_termination', 200]
seed = 1.0
generator = 'get_random_numbers_in_bounds'

variable_keys = [
                    [['soma', 'cm']],
                    [['soma', '0.5', 'pas', 'g']],
                    [['soma', '0.5', 'ih_fast', 'gbar']],
                    [['soma', '0.5', 'ih_slow', 'gbar']],
                    [['soma', '0.5', 'nat', 'gbar']],
                    [['soma', '0.5', 'kdr', 'gbar']],
                    [['soma', '0.5', 'ka', 'gbar']]
                 ]
bounds = {'lower_bounds': [0] * (len(variable_keys)), 'upper_bounds': [10] + [1.5] * (len(variable_keys)-1)}

model_dir = '../../model/cells/dapmodel_simpel.json'
mechanism_dir = '../../model/vclamp/stellate'
data_dir = '../../data/2015_08_26b/raw/rampIV/3.0(nA).csv'
data = pd.read_csv(data_dir)
errfun = 'rms'
fitfun = ['get_APamp', 'get_fAHPamp', 'get_DAPamp', 'get_vrest', 'get_v']
fitnessweights = [1, 1, 1, 1, 1]
args = {'threshold': -30}
#fitter = HodgkinHuxleyFitter(variable_keys, errfun, fitfun, fitnessweights,
#                 model_dir, mechanism_dir, data_dir, simulation_params={'celsius': 35}, args=args)
fitter = HodgkinHuxleyFitterPareto(variable_keys, errfun, fitfun, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params={'celsius': 35}, args=args)

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

algorithm_name = 'NSGA2'
algorithm_params = {'crossover_rate': 0.5, 'blx_alpha': 0.1,
                    'mutation_rate': 0.5, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.2}
normalize = False
optimization_params = {'variator': ['blend_crossover', 'gaussian_mutation']}

#optimization_params = None

save_dir = save_dir + '/' + algorithm_name + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

optimization_settings = OptimizationSettings(maximize, n_candidates, stop_criterion, seed, generator, bounds, fitter)
algorithm_settings = AlgorithmSettings(algorithm_name, algorithm_params, optimization_params, normalize, save_dir)

optimizer = OptimizerFactory().make_optimizer(optimization_settings, algorithm_settings)
optimizer.save(save_dir)
optimizer.optimize()