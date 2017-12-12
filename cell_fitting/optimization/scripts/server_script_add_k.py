import sys
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
from cell_fitting.optimization.scripts import optimize
from cell_fitting.optimization import generate_candidates
from cell_fitting.optimization.fitter.read_data import get_sweep_index_for_amp
from time import time
import os
import json


# parameters
save_dir = sys.argv[1]
process_number = int(sys.argv[2])
batch_size = int(sys.argv[3])
n_generations = 1000

variables = [
            [0.3, 2, [['soma', 'cm']]],
            [-100, -75, [['soma', '0.5', 'pas', 'e']]],
            [-30, -10, [['soma', '0.5', 'hcn_slow', 'ehcn']]],

            [0, 0.5, [['soma', '0.5', 'pas', 'g']]],
            [0, 0.5, [['soma', '0.5', 'nat', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'nap', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'hcn_slow', 'gbar']]],

            [-100, 0, [['soma', '0.5', 'nat', 'm_vh']]],
            [-100, 0, [['soma', '0.5', 'nat', 'h_vh']]],
            [-100, 0, [['soma', '0.5', 'nap', 'm_vh']]],
            [-100, 0, [['soma', '0.5', 'nap', 'h_vh']]],
            [-100, 0, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-100, 0, [['soma', '0.5', 'hcn_slow', 'n_vh']]],

            [1, 30, [['soma', '0.5', 'nat', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nat', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'nap', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nap', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'kdr', 'n_vs']]],
            [-30, -1, [['soma', '0.5', 'hcn_slow', 'n_vs']]],

            [0, 50, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0, 50, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0, 50, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0, 50, [['soma', '0.5', 'nap', 'h_tau_min']]],
            [0, 50, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 50, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]],

            [0, 100, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [0, 100, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [0, 100, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [0, 100, [['soma', '0.5', 'nap', 'h_tau_max']]],
            [0, 100, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0, 500, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]],

            [0, 1, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0, 1, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0, 1, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0, 1, [['soma', '0.5', 'nap', 'h_tau_delta']]],
            [0, 1, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 1, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]]
            ]
variables_extension = [
            [0, 0.5, [['soma', '0.5', 'ka', 'gbar']]],

            [-100, 0, [['soma', '0.5', 'ka', 'n_vh']]],
            [-100, 0, [['soma', '0.5', 'ka', 'l_vh']]],
            [1, 30, [['soma', '0.5', 'ka', 'n_vs']]],
            [-30, 1, [['soma', '0.5', 'ka', 'l_vs']]],

            [0, 50, [['soma', '0.5', 'ka', 'n_tau_min']]],
            [0, 50, [['soma', '0.5', 'ka', 'l_tau_min']]],
            [1, 100, [['soma', '0.5', 'ka', 'n_tau_max']]],
            [1, 100, [['soma', '0.5', 'ka', 'l_tau_max']]],
            [0, 1, [['soma', '0.5', 'ka', 'n_tau_delta']]],
            [0, 1, [['soma', '0.5', 'ka', 'l_tau_delta']]],
            ]
variables.extend(variables_extension)

variable_range_name = 'mean_std_6models'
save_dir_range = os.path.join('../../results/sensitivity_analysis/', 'variable_ranges')
with open(os.path.join(save_dir_range, variable_range_name + '.json'), 'r') as f:
    variables_init = json.load(f)
variables_init_extension = [
            [0.001, 0.05, [['soma', '0.5', 'ka', 'gbar']]],

            [-75, -30, [['soma', '0.5', 'ka', 'n_vh']]],
            [-75, -30, [['soma', '0.5', 'ka', 'l_vh']]],
            [1, 25, [['soma', '0.5', 'ka', 'n_vs']]],
            [-25, 1, [['soma', '0.5', 'ka', 'l_vs']]],

            [0, 10, [['soma', '0.5', 'ka', 'n_tau_min']]],
            [0, 10, [['soma', '0.5', 'ka', 'l_tau_min']]],
            [1, 50, [['soma', '0.5', 'ka', 'n_tau_max']]],
            [1, 50, [['soma', '0.5', 'ka', 'l_tau_max']]],
            [0, 1, [['soma', '0.5', 'ka', 'n_tau_delta']]],
            [0, 1, [['soma', '0.5', 'ka', 'l_tau_delta']]]
            ]
variables_init.extend(variables_init_extension)

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}
lower_bounds_init, upper_bounds_init, variable_keys_init = get_lowerbound_upperbound_keys(variables_init)
bounds_init = {'lower_bounds': list(lower_bounds_init), 'upper_bounds': list(upper_bounds_init)}

# read data
protocol = 'rampIV'
sweep_idx = get_sweep_index_for_amp(amp=3.1, protocol=protocol)
# data_read_dict = {'data_dir': '/home/cf/Phd/DAP-Project/cell_data/raw_data', 'cell_id': '2015_08_26b',
#                   'protocol': protocol, 'sweep_idx': sweep_idx, 'v_rest_shift': -16, 'file_type': 'dat'}
# protocol = 'hyperRampTester(1)'
# data_read_dict1 = {'data_dir': '/home/cf/Phd/DAP-Project/cell_data/raw_data', 'cell_id': '2013_12_11a',
#                   'protocol': protocol, 'sweep_idx': 0, 'v_rest_shift': -8, 'file_type': 'dat'}
# protocol = 'depoRampTester(1)'
# data_read_dict2 = {'data_dir': '/home/cf/Phd/DAP-Project/cell_data/raw_data', 'cell_id': '2013_12_11a',
#                   'protocol': protocol, 'sweep_idx': 0, 'v_rest_shift': -8, 'file_type': 'dat'}

data_read_dict = {'data_dir': '../../data/dat_files', 'cell_id': '2015_08_26b',
                  'protocol': protocol, 'sweep_idx': sweep_idx, 'v_rest_shift': -16, 'file_type': 'dat'}
protocol = 'hyperRampTester(3)'
data_read_dict1 = {'data_dir': '../../data/dat_files', 'cell_id': '2013_12_11a',
                  'protocol': protocol, 'sweep_idx': 0, 'v_rest_shift': -8, 'file_type': 'dat'}
protocol = 'depoRampTester(3)'
data_read_dict2 = {'data_dir': '../../data/dat_files', 'cell_id': '2013_12_11a',
                  'protocol': protocol, 'sweep_idx': 0, 'v_rest_shift': -8, 'file_type': 'dat'}

# dicts for fitting
fitter_params = {
                    #'name': 'HodgkinHuxleyFitter',
                    #'name': 'HodgkinHuxleyFitterAdaptive',
                    'name': 'HodgkinHuxleyFitterFitfunFromSet',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'model_dir': '../../model/cells/dapmodel_simpel.json',
                    'mechanism_dir': '../../model/channels/vavoulis',
                    'fitfun_names_per_data_set': [['v_and_diff_fAHP']],
                    #'fitfun_names_per_data_set': [['get_v'], ['get_fAHP_min'], ['get_fAHP_min']],
                    'fitnessweights_per_data_set': [[2, 1]],
                    #'fitnessweights_per_data_set': [[1], [10], [10]],
                    'data_read_dict_per_data_set': [data_read_dict, data_read_dict1, data_read_dict2],
                    'init_simulation_params': {'celsius': 35, 'onset': 200, 'v_init': -75},
                    #'init_simulation_params': {'celsius': 35, 'onset': 200, 'atol': 1e-5},
                    'args': {'max_fitness_error': 1000}
                }

optimization_settings_dict = {
    'maximize': False,
    'n_candidates': batch_size,
    'stop_criterion': ['generation_termination', n_generations],
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
# algorithm_settings_dict = {
#     'algorithm_name': 'PSO',
#     'algorithm_params': {'inertia': 0.4, 'cognitive_rate': 1.4, 'social_rate': 1.6},
#     'optimization_params': {},
#     'normalize': True,
#     'save_dir': os.path.join(save_dir, sys.argv[2])
# }
# algorithm_settings_dict = {
#     'algorithm_name': 'DEA',
#     'algorithm_params': {'num_selected': 335, 'tournament_size': 180, 'crossover_rate': 0.57,
#                          'mutation_rate': 0.52, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.21},
#     'optimization_params': {},
#     'normalize': True,
#     'save_dir': os.path.join(save_dir, sys.argv[2])
# }
# algorithm_settings_dict = {
#     'algorithm_name': 'SA',
#     'algorithm_params': {'temperature': 524.0, 'cooling_rate': 0.5, 'mutation_rate': 0.7,
#                          'gaussian_mean': 0.0, 'gaussian_stdev': 0.20},
#     'optimization_params': {},
#     'normalize': True,
#     'save_dir': os.path.join(save_dir, sys.argv[2])
# }
# algorithm_settings_dict = {
#     'algorithm_name': 'Nelder-Mead',
#     'algorithm_params': {},
#     'optimization_params': {},
#     'normalize': False,
#     'save_dir': os.path.join(save_dir, sys.argv[2])
# }
# algorithm_settings_dict = {
#     'algorithm_name': 'Random',
#     'algorithm_params': {},
#     'optimization_params': {},
#     'normalize': False,
#     'save_dir': os.path.join(save_dir, sys.argv[2])
# }
# algorithm_settings_dict = {
#     'algorithm_name': 'adam',
#     'algorithm_params': {},
#     'optimization_params': {},
#     'normalize': False,
#     'save_dir': os.path.join(save_dir, sys.argv[2])
# }

# generate initial candidates
init_candidates = generate_candidates(optimization_settings_dict['generator'],
                                      bounds_init['lower_bounds'],
                                      bounds_init['upper_bounds'],
                                      optimization_settings_dict['seed'] * process_number,
                                      optimization_settings_dict['n_candidates'])
optimization_settings_dict['extra_args']['init_candidates'] = init_candidates

# start optimization
optimize(optimization_settings_dict, algorithm_settings_dict)