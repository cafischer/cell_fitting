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
            [0.5, 2, [['soma', 'cm']]],
            [-95, -70, [['soma', '0.5', 'pas', 'e']]],

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
            [0.65, 0.75, [['soma', 'cm']]],
            [-87, -86, [['soma', '0.5', 'pas', 'e']]],

            [0.004, 0.006, [['soma', '0.5', 'pas', 'g']]],
            [0.08, 0.1, [['soma', '0.5', 'nat', 'gbar']]],
            [0.03, 0.05, [['soma', '0.5', 'kdr', 'gbar']]],
            [0.3, 0.5, [['soma', '0.5', 'nap', 'gbar']]],

            [-56, -54, [['soma', '0.5', 'nat', 'm_vh']]],
            [-82, -80, [['soma', '0.5', 'nat', 'h_vh']]],
            [-67, -65, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-34, -32, [['soma', '0.5', 'nap', 'm_vh']]],
            [-72, -70, [['soma', '0.5', 'nap', 'h_vh']]],

            [16, 18, [['soma', '0.5', 'nat', 'm_vs']]],
            [-24, -22, [['soma', '0.5', 'nat', 'h_vs']]],
            [18, 20, [['soma', '0.5', 'kdr', 'n_vs']]],
            [17, 19, [['soma', '0.5', 'nap', 'm_vs']]],
            [-14, -12, [['soma', '0.5', 'nap', 'h_vs']]],

            [0, 0.1, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0.4, 0.6, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0.5, 0.7, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 0.00001, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0.1, 0.3, [['soma', '0.5', 'nap', 'h_tau_min']]],

            [17, 19, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [18, 21, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [19, 22, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0.1, 0.4, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [8, 11, [['soma', '0.5', 'nap', 'h_tau_max']]],

            [0.3, 0.5, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0.6, 0.8, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0.4, 0.6, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0.01, 0.2, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0, 0.2, [['soma', '0.5', 'nap', 'h_tau_delta']]],
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
                    'data_dir': '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv',
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