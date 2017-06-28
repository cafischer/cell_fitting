import sys
sys.path.append("../../")
from optimization.helpers import get_lowerbound_upperbound_keys
from time import time
from new_optimization.scripts import optimize
from new_optimization import generate_initial_candidates
import os


# parameters
save_dir = sys.argv[1]

variables = [
            [0.7, 2, [['soma', 'cm']]],
            [-80, -60, [['soma', '0.5', 'pas', 'e']]],

            [0, 0.5, [['soma', '0.5', 'pas', 'g']]],
            [0, 0.5, [['soma', '0.5', 'nat', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'nap', 'gbar']]],

            [-100, 0, [['soma', '0.5', 'nat', 'm_vh']]],
            [-100, 0, [['soma', '0.5', 'nat', 'h_vh']]],
            [-100, 0, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-100, 0, [['soma', '0.5', 'nap', 'm_vh']]],
            [-100, 0, [['soma', '0.5', 'nap', 'h_vh']]],

            [1, 30, [['soma', '0.5', 'nat', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nat', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'kdr', 'n_vs']]],
            [1, 30, [['soma', '0.5', 'nap', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nap', 'h_vs']]],

            [0, 50, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0, 50, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0, 50, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 50, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0, 50, [['soma', '0.5', 'nap', 'h_tau_min']]],

            [0, 100, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [0, 100, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [0, 100, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0, 100, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [0, 100, [['soma', '0.5', 'nap', 'h_tau_max']]],

            [0, 10, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0, 10, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0, 10, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 10, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0, 10, [['soma', '0.5', 'nap', 'h_tau_delta']]],
            ]

variables_init = [
            [0.9, 1.1, [['soma', 'cm']]],
            [-75, -60, [['soma', '0.5', 'pas', 'e']]],

            [0.002, 0.005, [['soma', '0.5', 'pas', 'g']]],
            [0.06, 0.11, [['soma', '0.5', 'nat', 'gbar']]],
            [0.01, 0.04, [['soma', '0.5', 'kdr', 'gbar']]],
            [0.3, 0.45, [['soma', '0.5', 'nap', 'gbar']]],

            [-40, -34, [['soma', '0.5', 'nat', 'm_vh']]],
            [-68, -61, [['soma', '0.5', 'nat', 'h_vh']]],
            [-55, -49, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-18, -12, [['soma', '0.5', 'nap', 'm_vh']]],
            [-60, -54, [['soma', '0.5', 'nap', 'h_vh']]],

            [14, 18, [['soma', '0.5', 'nat', 'm_vs']]],
            [-24, -20, [['soma', '0.5', 'nat', 'h_vs']]],
            [14, 19, [['soma', '0.5', 'kdr', 'n_vs']]],
            [17, 21, [['soma', '0.5', 'nap', 'm_vs']]],
            [-15, -11, [['soma', '0.5', 'nap', 'h_vs']]],

            [0, 1, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0, 1, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0, 1, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 0.001, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0, 0.2, [['soma', '0.5', 'nap', 'h_tau_min']]],

            [14, 19, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [19, 23, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [17, 22, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0.01, 1, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [8, 13, [['soma', '0.5', 'nap', 'h_tau_max']]],

            [0.2, 0.7, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0.2, 0.7, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0.3, 0.8, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0.05, 0.3, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0.1, 0.5, [['soma', '0.5', 'nap', 'h_tau_delta']]],
            ]


lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}
lower_bounds_init, upper_bounds_init, variable_keys_init = get_lowerbound_upperbound_keys(variables_init)
bounds_init = {'lower_bounds': list(lower_bounds_init), 'upper_bounds': list(upper_bounds_init)}

fitter_params = {
                    'name': 'HodgkinHuxleyFitterAdaptive',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'fitfun_names': ['get_v'],
                    'fitnessweights': [1],
                    'model_dir': '../../model/cells/dapmodel_simpel.json',
                    'mechanism_dir': '../../model/channels/vavoulis',
                    'data_dir': '../../data/2015_08_06d/vrest-60/rampIV/3.5(nA).csv',
                    'simulation_params': {'celsius': 35, 'onset': 200, 'atol': 1e-8, 'continuous': True,
                                          'discontinuities': None, 'interpolate': True},
                    'args': {}
                }

optimization_settings_dict = {
    'maximize': False,
    'n_candidates': 100000,
    'stop_criterion': ['generation_termination', 1000],
    'seed': time(),
    'generator': 'get_random_numbers_in_bounds',
    'bounds': bounds,
    'fitter_params': fitter_params,
    'extra_args': {}
}

algorithm_settings_dict = {
    'algorithm_name': 'L-BFGS-B',
    'algorithm_params': {},
    'optimization_params': {},
    'normalize': False,
    'save_dir': os.path.join(save_dir, sys.argv[2])
}

# generate initial candidates
init_candidates = generate_initial_candidates(optimization_settings_dict['generator'],
                            bounds_init['lower_bounds'],
                            bounds_init['upper_bounds'],
                            optimization_settings_dict['seed'],
                            optimization_settings_dict['n_candidates'])

# choose right candidate
batch_size = sys.argv[3]
optimization_settings_dict['extra_args']['init_candidates'] = init_candidates[int(sys.argv[2])*int(batch_size):
                                                               (int(sys.argv[2])+1)*int(batch_size)]
#optimization_settings_dict['extra_args']['init_candidates'] = init_candidates[0:1]

# start optimization
optimize(optimization_settings_dict, algorithm_settings_dict)