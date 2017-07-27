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
            [0.4, 2, [['soma', 'cm']]],
            [-95, -70, [['soma', '0.5', 'pas', 'e']]],
            [-30, -10, [['soma', '0.5', 'hcn_slow', 'ehcn']]],

            [0, 0.5, [['soma', '0.5', 'pas', 'g']]],
            [0, 0.5, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 1.0, [['soma', '0.5', 'nap', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'hcn_slow', 'gbar']]],

            [0, 6, [['soma', '0.5', 'kdr', 'n_pow']]],
            [0, 6, [['soma', '0.5', 'nap', 'm_pow']]],
            [0, 6, [['soma', '0.5', 'nap', 'h_pow']]],
            [0, 6, [['soma', '0.5', 'hcn_slow', 'n_pow']]],

            [-100, 0, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-100, 0, [['soma', '0.5', 'nap', 'm_vh']]],
            [-100, 0, [['soma', '0.5', 'nap', 'h_vh']]],
            [-100, 0, [['soma', '0.5', 'hcn_slow', 'n_vh']]],

            [1, 30, [['soma', '0.5', 'kdr', 'n_vs']]],
            [1, 30, [['soma', '0.5', 'nap', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nap', 'h_vs']]],
            [-30, -1, [['soma', '0.5', 'hcn_slow', 'n_vs']]],

            [0, 50, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 50, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0, 50, [['soma', '0.5', 'nap', 'h_tau_min']]],
            [0, 50, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]],

            [0, 100, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0, 100, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [0, 100, [['soma', '0.5', 'nap', 'h_tau_max']]],
            [0, 500, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]],

            [0, 10, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 10, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0, 10, [['soma', '0.5', 'nap', 'h_tau_delta']]],
            [0, 10, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
            ]

variables_init = [
            [0.5, 0.8, [['soma', 'cm']]],
            [-89, -84, [['soma', '0.5', 'pas', 'e']]],
            [-30, -24, [['soma', '0.5', 'hcn_slow', 'ehcn']]],

            [0.0005, 0.0015, [['soma', '0.5', 'pas', 'g']]],
            [0.001, 0.015, [['soma', '0.5', 'kdr', 'gbar']]],
            [0.07, 0.25, [['soma', '0.5', 'nap', 'gbar']]],
            [0.00004, 0.0003, [['soma', '0.5', 'hcn_slow', 'gbar']]],

            [3.9, 4.1, [['soma', '0.5', 'kdr', 'n_pow']]],
            [2.9, 3.1, [['soma', '0.5', 'nap', 'm_pow']]],
            [0.9, 1.1, [['soma', '0.5', 'nap', 'h_pow']]],
            [0.9, 1.1, [['soma', '0.5', 'hcn_slow', 'n_pow']]],

            [-70, -65, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-35, -30, [['soma', '0.5', 'nap', 'm_vh']]],
            [-75, -58, [['soma', '0.5', 'nap', 'h_vh']]],
            [-87, -77, [['soma', '0.5', 'hcn_slow', 'n_vh']]],

            [17, 21, [['soma', '0.5', 'kdr', 'n_vs']]],
            [11, 19, [['soma', '0.5', 'nap', 'm_vs']]],
            [-16, -12, [['soma', '0.5', 'nap', 'h_vs']]],
            [-21, -17, [['soma', '0.5', 'hcn_slow', 'n_vs']]],

            [0.2, 0.7, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 0.001, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0, 0.2, [['soma', '0.5', 'nap', 'h_tau_min']]],
            [2, 7, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]],

            [19, 23, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0.1, 0.5, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [6, 11, [['soma', '0.5', 'nap', 'h_tau_max']]],
            [125, 141, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]],

            [0.5, 0.95, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 0.4, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0.1, 0.5, [['soma', '0.5', 'nap', 'h_tau_delta']]],
            [0, 0.4, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
            ]


lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}
lower_bounds_init, upper_bounds_init, variable_keys_init = get_lowerbound_upperbound_keys(variables_init)
bounds_init = {'lower_bounds': list(lower_bounds_init), 'upper_bounds': list(upper_bounds_init)}

fitter_params = {
                    'name': 'HodgkinHuxleyFitterSeveralDataSeveralFitfuns',
                    #'name': 'HodgkinHuxleyFitter',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'fitfun_names': [['get_v', 'get_DAP'], ['get_v'], ['get_v'], ['get_n_spikes']],
                    'model_dir': '../../model/cells/dapmodel_simpel.json',
                    'mechanism_dir': '../../model/channels/vavoulis',
                    # 'fitnessweights': [1, 1000],
                    # 'data_dir': '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv',
                    # 'simulation_params': {'celsius': 35, 'onset': 200},
                    'fitnessweights': [[100, 10], [1], [1], [11]],
                    'data_dirs': [
                                  '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv',
                                  '../../data/2015_08_26b/vrest-75/IV/-0.1(nA).csv',
                                  '../../data/2015_08_26b/vrest-75/IV/0.2(nA).csv',
                                  '../../data/2015_08_26b/vrest-75/IV/0.4(nA).csv'
                                  ],
                    'simulation_params': {'celsius': 35, 'onset': 200},
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