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
n_generations = 500

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

            [-100, 0, [['soma', '0.5', 'nat', 'm_tau_vh']]],
            [-100, 0, [['soma', '0.5', 'nat', 'h_tau_vh']]],
            [-100, 0, [['soma', '0.5', 'nap', 'm_tau_vh']]],
            [-100, 0, [['soma', '0.5', 'nap', 'h_tau_vh']]],
            [-100, 0, [['soma', '0.5', 'kdr', 'n_tau_vh']]],
            [-100, 0, [['soma', '0.5', 'hcn_slow', 'n_tau_vh']]],

            [1, 30, [['soma', '0.5', 'nat', 'm_tau_vs']]],
            [-30, -1, [['soma', '0.5', 'nat', 'h_tau_vs']]],
            [1, 30, [['soma', '0.5', 'nap', 'm_tau_vs']]],
            [-30, -1, [['soma', '0.5', 'nap', 'h_tau_vs']]],
            [1, 30, [['soma', '0.5', 'kdr', 'n_tau_vs']]],
            [-30, -1, [['soma', '0.5', 'hcn_slow', 'n_tau_vs']]],

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

new_variables = [
                [-100, 0, [['soma', '0.5', 'nat', 'm_tau_vh']]],
                [-100, 0, [['soma', '0.5', 'nat', 'h_tau_vh']]],
                [-100, 0, [['soma', '0.5', 'nap', 'm_tau_vh']]],
                [-100, 0, [['soma', '0.5', 'nap', 'h_tau_vh']]],
                [-100, 0, [['soma', '0.5', 'kdr', 'n_tau_vh']]],
                [-100, 0, [['soma', '0.5', 'hcn_slow', 'n_tau_vh']]],

                [1, 30, [['soma', '0.5', 'nat', 'm_tau_vs']]],
                [-30, -1, [['soma', '0.5', 'nat', 'h_tau_vs']]],
                [1, 30, [['soma', '0.5', 'nap', 'm_tau_vs']]],
                [-30, -1, [['soma', '0.5', 'nap', 'h_tau_vs']]],
                [1, 30, [['soma', '0.5', 'kdr', 'n_tau_vs']]],
                [-30, -1, [['soma', '0.5', 'hcn_slow', 'n_tau_vs']]]
                ]

variable_range_name = 'mean_std_1order_of_mag_model2'
save_dir_range = os.path.join('../../results/sensitivity_analysis/', 'variable_ranges')
with open(os.path.join(save_dir_range, variable_range_name + '.json'), 'r') as f:
    variables_init = json.load(f)
variables_init = variables_init[:20] + variables_init[8:20] + variables_init[20:]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}
lower_bounds_init, upper_bounds_init, variable_keys_init = get_lowerbound_upperbound_keys(variables_init)
bounds_init = {'lower_bounds': list(lower_bounds_init), 'upper_bounds': list(upper_bounds_init)}

# read data
protocol = 'rampIV'
sweep_idx = get_sweep_index_for_amp(amp=3.1, protocol=protocol)
data_read_dict = {'data_dir': '../../data/dat_files', 'cell_id': '2015_08_26b',
                  'protocol': protocol, 'sweep_idx': sweep_idx, 'v_rest_shift': -16, 'file_type': 'dat'}

protocol = 'IV'
sweep_idx = get_sweep_index_for_amp(amp=-0.15, protocol=protocol)
data_read_dict1 = {'data_dir': '../../data/dat_files', 'cell_id': '2015_08_26b',
                  'protocol': protocol, 'sweep_idx': sweep_idx, 'v_rest_shift': -16, 'file_type': 'dat'}

protocol = 'IV'
sweep_idx = get_sweep_index_for_amp(amp=1.0, protocol=protocol)
data_read_dict2 = {'data_dir': '../../data/dat_files', 'cell_id': '2015_08_26b',
                  'protocol': protocol, 'sweep_idx': sweep_idx, 'v_rest_shift': -16, 'file_type': 'dat'}

# dicts for fitting
fitter_params = {
                #'name': 'HodgkinHuxleyFitter',
                'name': 'HodgkinHuxleyFitterFitfunFromSet',
                'variable_keys': variable_keys,
                'errfun_name': 'rms',
                'model_dir': '../../model/cells/dapmodel_simpel.json',
                'mechanism_dir': '../../model/channels/vavoulis_independent_tau',
                #'fitfun_names_per_data_set': [['get_v']],
                #'fitnessweights_per_data_set': [[1]],
                # 'data_read_dict_per_data_set': [data_read_dict],
                'fitfun_names_per_data_set': [['get_v'], ['get_v'], ['get_n_spikes']],
                'fitnessweights_per_data_set': [[1], [1], [1]],
                'data_read_dict_per_data_set': [data_read_dict, data_read_dict1, data_read_dict2],
                'init_simulation_params': {'celsius': 35, 'onset': 200, 'v_init': -75},
                'args': {'max_fitness_error': 100}
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