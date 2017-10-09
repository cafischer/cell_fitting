import sys
sys.path.append("../../")
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
from time import time
from cell_fitting.new_optimization.scripts import optimize
from cell_fitting.new_optimization import generate_initial_candidates
import os


# parameters
save_dir = sys.argv[1]
process_number = int(sys.argv[2])
batch_size = int(sys.argv[3])

variables = [
            [0.3, 2, [['soma', 'cm']]],
            [-100, -75, [['soma', '0.5', 'pas', 'e']]],
            [-30, -10, [['soma', '0.5', 'hcn_slow', 'ehcn']]],

            [0, 0.5, [['soma', '0.5', 'pas', 'g']]],
            [0, 0.5, [['soma', '0.5', 'nat', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'hcn_slow', 'gbar']]],

            [0, 6, [['soma', '0.5', 'nat', 'm_pow']]],
            [0, 6, [['soma', '0.5', 'nat', 'h_pow']]],
            [0, 6, [['soma', '0.5', 'kdr', 'n_pow']]],
            [0, 6, [['soma', '0.5', 'hcn_slow', 'n_pow']]],

            [-100, 0, [['soma', '0.5', 'nat', 'm_vh']]],
            [-100, 0, [['soma', '0.5', 'nat', 'h_vh']]],
            [-100, 0, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-100, 0, [['soma', '0.5', 'hcn_slow', 'n_vh']]],

            [1, 30, [['soma', '0.5', 'nat', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nat', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'kdr', 'n_vs']]],
            [-30, -1, [['soma', '0.5', 'hcn_slow', 'n_vs']]],

            [0, 50, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0, 50, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0, 50, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 50, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]],

            [0, 100, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [0, 100, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [0, 100, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0, 500, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]],

            [0, 10, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0, 10, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0, 10, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 10, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
            ]

variables_init = [
            [0.7, 0.9, [['soma', 'cm']]],
            [-96, -91, [['soma', '0.5', 'pas', 'e']]],
            [-26, -23, [['soma', '0.5', 'hcn_slow', 'ehcn']]],

            [0.00005, 0.0001, [['soma', '0.5', 'pas', 'g']]],
            [0.06, 0.1, [['soma', '0.5', 'nat', 'gbar']]],
            [0.003, 0.01, [['soma', '0.5', 'kdr', 'gbar']]],
            [0.00006, 0.0004, [['soma', '0.5', 'hcn_slow', 'gbar']]],

            [1.6, 1.9, [['soma', '0.5', 'nat', 'm_pow']]],
            [2.9, 3.2, [['soma', '0.5', 'nat', 'h_pow']]],
            [3.5, 3.9, [['soma', '0.5', 'kdr', 'n_pow']]],
            [2.0, 2.4, [['soma', '0.5', 'hcn_slow', 'n_pow']]],

            [-60, -55, [['soma', '0.5', 'nat', 'm_vh']]],
            [-80, -76, [['soma', '0.5', 'nat', 'h_vh']]],
            [-61, -57, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-72, -68, [['soma', '0.5', 'hcn_slow', 'n_vh']]],

            [1, 3, [['soma', '0.5', 'nat', 'm_vs']]],
            [-22, -18, [['soma', '0.5', 'nat', 'h_vs']]],
            [3, 7, [['soma', '0.5', 'kdr', 'n_vs']]],
            [-20, -16, [['soma', '0.5', 'hcn_slow', 'n_vs']]],

            [0, 0.0001, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0.2, 0.6, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0.2, 0.6, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [2, 6, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]],

            [5, 9, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [8, 12, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [23, 28, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [132, 136, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]],

            [0.9, 1.1, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0.3, 0.6, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0.2, 0.5, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 0.005, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
            ]


lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}
lower_bounds_init, upper_bounds_init, variable_keys_init = get_lowerbound_upperbound_keys(variables_init)
bounds_init = {'lower_bounds': list(lower_bounds_init), 'upper_bounds': list(upper_bounds_init)}

fitter_params = {
                    #'name': 'HodgkinHuxleyFitterSeveralDataSeveralFitfuns',
                    'name': 'HodgkinHuxleyFitter',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'model_dir': '../../model/cells/dapmodel_simpel.json',
                    'mechanism_dir': '../../model/channels/vavoulis',
                    'fitfun_names': ['get_v'],
                    'fitnessweights': [1],
                    'data_dir': '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv',
                    'simulation_params': {'celsius': 35, 'onset': 200},
                    # 'fitfun_names': [['get_v'], ['get_v']],
                    # 'fitnessweights': [[1], [1]],
                    # 'data_dirs': [
                    #               '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv',
                    #               '../../data/2015_08_26b/vrest-75/IV/0.3(nA).csv'
                    #               ],
                    # 'simulation_params': {'celsius': 35, 'onset': 200},
                    'args': {}
                }

optimization_settings_dict = {
    'maximize': False,
    'n_candidates': batch_size,
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
                            optimization_settings_dict['seed'] * process_number,
                            optimization_settings_dict['n_candidates'])

# choose right candidate
optimization_settings_dict['extra_args']['init_candidates'] = init_candidates

# start optimization
optimize(optimization_settings_dict, algorithm_settings_dict)