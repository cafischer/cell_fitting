import sys
sys.path.append("../../")
from optimization.helpers import get_lowerbound_upperbound_keys
from time import time
from new_optimization.scripts import optimize_hyperparameters
from new_optimization import generate_initial_candidates
import os

# parameters
#save_dir = '../../results/'+sys.argv[1]+'/'
save_dir = '../../results/test/'

variables = [
            [0.7, 1.5, [['soma', 'cm']]],
            [-100, -50, [['soma', '0.5', 'pas', 'e']]],

            [0, 0.001, [['soma', '0.5', 'pas', 'g']]],
            [0, 2, [['soma', '0.5', 'nap_act', 'gbar']]],
            [0, 2, [['soma', '0.5', 'nat', 'gbar']]],
            [0, 2, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 2, [['soma', '0.5', 'ka', 'gbar']]],
            [0, 0.1, [['soma', '0.5', 'hcn_fast', 'gbar']]],

            [-40, 20, [['soma', '0.5', 'nap_act', 'm_vh']]],
            [-100, 20, [['soma', '0.5', 'nat', 'm_vh']]],
            [-100, -10, [['soma', '0.5', 'nat', 'h_vh']]],
            [-100, -10, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-100, -10, [['soma', '0.5', 'ka', 'n_vh']]],
            [-100, -10, [['soma', '0.5', 'ka', 'l_vh']]],
            [-100, -10, [['soma', '0.5', 'hcn_fast', 'n_vh']]],

            [1, 30, [['soma', '0.5', 'nap_act', 'm_vs']]],
            [1, 30, [['soma', '0.5', 'nat', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nat', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'kdr', 'n_vs']]],
            [1, 30, [['soma', '0.5', 'ka', 'n_vs']]],
            [1, 30, [['soma', '0.5', 'ka', 'l_vs']]],
            [-30, -1, [['soma', '0.5', 'hcn_fast', 'n_vs']]],

            [0, 50, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0, 50, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0, 50, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 50, [['soma', '0.5', 'ka', 'n_tau_min']]],
            [0, 50, [['soma', '0.5', 'ka', 'l_tau_min']]],
            [0, 50, [['soma', '0.5', 'hcn_fast', 'n_tau_min']]],

            [0.001, 50, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [0.001, 50, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [0.001, 50, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0.001, 50, [['soma', '0.5', 'ka', 'n_tau_max']]],
            [0.001, 50, [['soma', '0.5', 'ka', 'l_tau_max']]],
            [10, 500, [['soma', '0.5', 'hcn_fast', 'n_tau_max']]],

            [0, 5, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0, 5, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0, 5, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 5, [['soma', '0.5', 'ka', 'n_tau_delta']]],
            [0, 5, [['soma', '0.5', 'ka', 'l_tau_delta']]],
            [0, 5, [['soma', '0.5', 'hcn_fast', 'n_tau_delta']]]
            ]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}

fitter_params = {
                    'name': 'HodgkinHuxleyFitterSeveralData',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'fitfun_names': ['get_v'],
                    'fitnessweights': [1],
                    'model_dir': '../../model/cells/dapmodel_simpel.json',
                    'mechanism_dir': '../../model/channels/vavoulis',
                    'data_dirs': ['../../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(3)/0(nA).csv',
                                  '../../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(21)/0(nA).csv'],
                    'simulation_params': {'celsius': 35, 'onset': 200},
                    'args': {}
                }

optimization_settings_dict = {
    'maximize': False,
    'n_candidates': 1,
    'stop_criterion': ['generation_termination', 3],
    'seed': time(),
    'generator': 'get_random_numbers_in_bounds',
    'bounds': bounds,
    'fitter_params': fitter_params,
    'extra_args': {}
}

algorithm_settings_dict = {
    'algorithm_name': 'adam',
    'algorithm_params': {},
    'optimization_params': {},
    'normalize': False,
    'save_dir': os.path.join(save_dir, '0') #sys.argv[2])
}

hyperparameter_dict = {
    'parameter_names': ['step_rate', 'decay_mom1', 'decay_mom2'],
    'lower_bounds': [1e-10, 0, 0],
    'upper_bounds': [1, 1, 1],
    'seed': time(),
    'n_samples': 1#int(sys.argv[3])
}

# generate initial candidates
init_candidates = generate_initial_candidates(optimization_settings_dict['generator'],
                            optimization_settings_dict['bounds']['lower_bounds'],
                            optimization_settings_dict['bounds']['upper_bounds'],
                            optimization_settings_dict['seed'],
                            optimization_settings_dict['n_candidates'])

# choose right candidate
batch_size = 1
#optimization_settings_dict['extra_args']['init_candidates'] = init_candidates[int(sys.argv[2])*batch_size:
#                                                                             (int(sys.argv[2])+1)*batch_size]
optimization_settings_dict['extra_args']['init_candidates'] = init_candidates[0:1]

# start optimization
optimize_hyperparameters(hyperparameter_dict, optimization_settings_dict, algorithm_settings_dict)