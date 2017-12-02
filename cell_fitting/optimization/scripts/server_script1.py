import sys
sys.path.append("../../")
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
from cell_fitting.optimization.scripts import optimize
from cell_fitting.optimization import generate_candidates
from cell_fitting.optimization.fitter.read_data import get_sweep_index_for_amp
import os


# parameters
save_dir = sys.argv[1]
process_number = int(sys.argv[2])
batch_size = int(sys.argv[3])


# vs not too small |vs| > 1 otherwise overflow in exp
# delta e [0, 1]
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

            [0, 1, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0, 1, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0, 1, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 1, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
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

            [0.9, 1.0, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0.3, 0.6, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0.2, 0.5, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 0.005, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
            ]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}
lower_bounds_init, upper_bounds_init, variable_keys_init = get_lowerbound_upperbound_keys(variables_init)
bounds_init = {'lower_bounds': list(lower_bounds_init), 'upper_bounds': list(upper_bounds_init)}

# read data
protocol = 'rampIV'
sweep_idx = get_sweep_index_for_amp(amp=0.2, protocol=protocol)
#data_read_dict = {'data_dir': '../../data/cell_csv_data', 'cell_id': '2015_08_26b', 'protocol': protocol, 'sweep_idx': sweep_idx,
#                  'v_rest_shift': -16, 'file_type': 'csv'}
data_read_dict = {'data_dir': '/home/cf/Phd/DAP-Project/cell_data/raw_data', 'cell_id': '2015_08_26b',
                  'protocol': protocol, 'sweep_idx': sweep_idx, 'v_rest_shift': -16, 'file_type': 'dat'}

# dicts for fitting
fitter_params = {
                    'name': 'HodgkinHuxleyFitter',
                    #'name': 'HodgkinHuxleyFitterAdaptive',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'model_dir': '../../model/cells/dapmodel_simpel.json',
                    'mechanism_dir': '../../model/channels/vavoulis',
                    'fitfun_names_per_data_set': [['get_v']],
                    'fitnessweights_per_data_set': [[1]],
                    'data_read_dict_per_data_set': [data_read_dict],
                    'init_simulation_params': {'celsius': 35, 'onset': 200},
                    #'init_simulation_params': {'celsius': 35, 'onset': 200, 'atol': 1e-5},
                    'args': {'max_fitness_error': 100000}
                }

optimization_settings_dict = {
    'maximize': False,
    'n_candidates': batch_size,
    'stop_criterion': ['generation_termination', 2],
    'seed': time(),
    'generator': 'get_random_numbers_in_bounds',
    'bounds': bounds,
    'fitter_params': fitter_params,
    'extra_args': {}
}

# algorithm_settings_dict = {
#     'algorithm_name': 'L-BFGS-B',
#     'algorithm_params': {},
#     'optimization_params': {},
#     'normalize': False,
#     'save_dir': os.path.join(save_dir, sys.argv[2])
# }
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
algorithm_settings_dict = {
    'algorithm_name': 'adam',
    'algorithm_params': {},
    'optimization_params': {},
    'normalize': False,
    'save_dir': os.path.join(save_dir, sys.argv[2])
}

# generate initial candidates
init_candidates = generate_candidates(optimization_settings_dict['generator'],
                                      bounds_init['lower_bounds'],
                                      bounds_init['upper_bounds'],
                                      optimization_settings_dict['seed'] * process_number,
                                      optimization_settings_dict['n_candidates'])
optimization_settings_dict['extra_args']['init_candidates'] = init_candidates

# start optimization
optimize(optimization_settings_dict, algorithm_settings_dict)