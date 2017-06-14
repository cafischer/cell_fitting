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
            [0.1, 1.5, [['soma', 'cm']]],
            [-90, -70, [['soma', '0.5', 'pas', 'e']]],
            [-115, -80, [['soma', '0.5', 'ek']]],
            [55, 65, [['soma', '0.5', 'ena']]],
            [80, 100, [['soma', '0.5', 'eca']]],

            [0, 0.001, [['soma', '0.5', 'pas', 'g']]],
            [0, 0.5, [['soma', '0.5', 'nap', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'nat', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'ka', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'cah2', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'hcn2', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'nap_v', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'nat_v', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'kdr_v', 'gbar']]],

            [-90, 0, [['soma', '0.5', 'nap', 'm_vh']]],
            [-90, 0, [['soma', '0.5', 'nat', 'm_vh']]],
            [-90, 0, [['soma', '0.5', 'nat', 'h_vh']]],
            [-90, 10, [['soma', '0.5', 'ka', 'm_vh']]],
            [-90, 0, [['soma', '0.5', 'ka', 'h_vh']]],
            [-90, 0, [['soma', '0.5', 'cah2', 'm_vh']]],
            [-90, 0, [['soma', '0.5', 'cah2', 'h_vh']]],
            [-90, 0, [['soma', '0.5', 'hcn2', 'h_vh']]],
            [-90, 0, [['soma', '0.5', 'nat_v', 'm_vh']]],
            [-90, 0, [['soma', '0.5', 'nat_v', 'h_vh']]],
            [-90, 0, [['soma', '0.5', 'nap_v', 'm_vh']]],
            [-90, 0, [['soma', '0.5', 'nap_v', 'h_vh']]],
            [-90, 0, [['soma', '0.5', 'kdr_v', 'm_vh']]],

            [1, 30, [['soma', '0.5', 'nap', 'm_vs']]],
            [1, 30, [['soma', '0.5', 'nat', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nat', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'ka', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'ka', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'cah2', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'cah2', 'h_vs']]],
            [-30, -1, [['soma', '0.5', 'hcn2', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'nat_v', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nat_v', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'nap_v', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nap_v', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'kdr_v', 'm_vs']]],

            [0, 5, [['soma', '0.5', 'nat', 'h_a']]],
            [0, 1, [['soma', '0.5', 'nat', 'h_b']]],
            [-90, 0, [['soma', '0.5', 'nat', 'h_c']]],
            [30, 80, [['soma', '0.5', 'nat', 'h_d']]],
            [0, 50, [['soma', '0.5', 'ka', 'm_tau_min']]],
            [0, 100, [['soma', '0.5', 'ka', 'm_tau_max']]],
            [0, 5, [['soma', '0.5', 'ka', 'm_tau_delta']]],
            [0, 1500, [['soma', '0.5', 'ka', 'h_tau_min']]],
            [100, 2000, [['soma', '0.5', 'ka', 'h_tau_max']]],
            [0, 5, [['soma', '0.5', 'ka', 'h_tau_delta']]],
            [0, 50, [['soma', '0.5', 'cah2', 'm_tau_min']]],
            [0, 200, [['soma', '0.5', 'cah2', 'm_tau_max']]],
            [0, 5, [['soma', '0.5', 'cah2', 'm_tau_delta']]],
            [0, 500, [['soma', '0.5', 'cah2', 'h_tau_min']]],
            [0, 1000, [['soma', '0.5', 'cah2', 'h_tau_max']]],
            [0, 5, [['soma', '0.5', 'cah2', 'h_tau_delta']]],
            [0, 100, [['soma', '0.5', 'hcn2', 'h_tau_min']]],
            [1, 500, [['soma', '0.5', 'hcn2', 'h_tau_max']]],
            [0, 5, [['soma', '0.5', 'hcn2', 'h_tau_delta']]],

            [0, 50, [['soma', '0.5', 'nat_v', 'm_tau_min']]],
            [0, 500, [['soma', '0.5', 'nat_v', 'h_tau_min']]],
            [0, 50, [['soma', '0.5', 'nap_v', 'm_tau_min']]],
            [0, 500, [['soma', '0.5', 'nap_v', 'h_tau_min']]],
            [0, 50, [['soma', '0.5', 'kdr_v', 'm_tau_min']]],

            [0, 100, [['soma', '0.5', 'nat_v', 'm_tau_max']]],
            [0, 500, [['soma', '0.5', 'nat_v', 'h_tau_max']]],
            [0, 100, [['soma', '0.5', 'nap_v', 'm_tau_max']]],
            [0, 500, [['soma', '0.5', 'nap_v', 'h_tau_max']]],
            [0, 100, [['soma', '0.5', 'kdr_v', 'm_tau_max']]],

            [0, 5, [['soma', '0.5', 'nat_v', 'm_tau_delta']]],
            [0, 5, [['soma', '0.5', 'nat_v', 'h_tau_delta']]],
            [0, 5, [['soma', '0.5', 'nap_v', 'm_tau_delta']]],
            [0, 5, [['soma', '0.5', 'nap_v', 'h_tau_delta']]],
            [0, 5, [['soma', '0.5', 'kdr_v', 'm_tau_delta']]],
            ]

variables_init = [
            [1.1, 1.3, [['soma', 'cm']]],
            [-85, -76, [['soma', '0.5', 'pas', 'e']]],
            [-110, -108, [['soma', '0.5', 'ek']]],
            [60, 60, [['soma', '0.5', 'ena']]],
            [90, 90, [['soma', '0.5', 'eca']]],

            [0, 0.0001, [['soma', '0.5', 'pas', 'g']]],
            [0.0005, 0.003, [['soma', '0.5', 'nap', 'gbar']]],
            [0.005, 0.04, [['soma', '0.5', 'nat', 'gbar']]],
            [0.005, 0.05, [['soma', '0.5', 'ka', 'gbar']]],
            [0.001, 0.02, [['soma', '0.5', 'cah2', 'gbar']]],
            [0, 0.0001, [['soma', '0.5', 'hcn2', 'gbar']]],
            [0, 0.001, [['soma', '0.5', 'nap_v', 'gbar']]],
            [0, 0.001, [['soma', '0.5', 'nat_v', 'gbar']]],
            [0, 0.001, [['soma', '0.5', 'kdr_v', 'gbar']]],

            [-50, -44, [['soma', '0.5', 'nap', 'm_vh']]],
            [-43, -36, [['soma', '0.5', 'nat', 'm_vh']]],
            [-71, -64, [['soma', '0.5', 'nat', 'h_vh']]],
            [-11, -6, [['soma', '0.5', 'ka', 'm_vh']]],
            [-70, -60, [['soma', '0.5', 'ka', 'h_vh']]],
            [-24, -16, [['soma', '0.5', 'cah2', 'm_vh']]],
            [-61, -54, [['soma', '0.5', 'cah2', 'h_vh']]],
            [-73, -60, [['soma', '0.5', 'hcn2', 'h_vh']]],
            [-45, -35, [['soma', '0.5', 'nat_v', 'm_vh']]],
            [-73, -62, [['soma', '0.5', 'nat_v', 'h_vh']]],
            [-25, -12, [['soma', '0.5', 'nap_v', 'm_vh']]],
            [-65, -54, [['soma', '0.5', 'nap_v', 'h_vh']]],
            [-55, -45, [['soma', '0.5', 'kdr_v', 'm_vh']]],

            [3, 7, [['soma', '0.5', 'nap', 'm_vs']]],
            [5, 9, [['soma', '0.5', 'nat', 'm_vs']]],
            [-8, -4, [['soma', '0.5', 'nat', 'h_vs']]],
            [11, 18, [['soma', '0.5', 'ka', 'm_vs']]],
            [-12, -4, [['soma', '0.5', 'ka', 'h_vs']]],
            [1, 8, [['soma', '0.5', 'cah2', 'm_vs']]],
            [-15, -8, [['soma', '0.5', 'cah2', 'h_vs']]],
            [-16, -9, [['soma', '0.5', 'hcn2', 'h_vs']]],
            [10, 18, [['soma', '0.5', 'nat_v', 'm_vs']]],
            [-26, -15, [['soma', '0.5', 'nat_v', 'h_vs']]],
            [12, 24, [['soma', '0.5', 'nap_v', 'm_vs']]],
            [-18, -8, [['soma', '0.5', 'nap_v', 'h_vs']]],
            [10, 21, [['soma', '0.5', 'kdr_v', 'm_vs']]],

            [0.2, 0.5, [['soma', '0.5', 'nat', 'h_a']]],
            [0, 0.005, [['soma', '0.5', 'nat', 'h_b']]],
            [-41, -35, [['soma', '0.5', 'nat', 'h_c']]],
            [48, 55, [['soma', '0.5', 'nat', 'h_d']]],
            [0, 0.001, [['soma', '0.5', 'ka', 'm_tau_min']]],
            [2, 9, [['soma', '0.5', 'ka', 'm_tau_max']]],
            [0, 0.5, [['soma', '0.5', 'ka', 'm_tau_delta']]],
            [800, 1200, [['soma', '0.5', 'ka', 'h_tau_min']]],
            [800, 1200, [['soma', '0.5', 'ka', 'h_tau_max']]],
            [0, 0.5, [['soma', '0.5', 'ka', 'h_tau_delta']]],
            [0, 6, [['soma', '0.5', 'cah2', 'm_tau_min']]],
            [3, 12, [['soma', '0.5', 'cah2', 'm_tau_max']]],
            [0, 1.5, [['soma', '0.5', 'cah2', 'm_tau_delta']]],
            [220, 300, [['soma', '0.5', 'cah2', 'h_tau_min']]],
            [290, 370, [['soma', '0.5', 'cah2', 'h_tau_max']]],
            [0, 0.4, [['soma', '0.5', 'cah2', 'h_tau_delta']]],
            [50, 90, [['soma', '0.5', 'hcn2', 'h_tau_min']]],
            [140, 200, [['soma', '0.5', 'hcn2', 'h_tau_max']]],
            [0, 0.5, [['soma', '0.5', 'hcn2', 'h_tau_delta']]],

            [0, 2, [['soma', '0.5', 'nat_v', 'm_tau_min']]],
            [0, 4, [['soma', '0.5', 'nat_v', 'h_tau_min']]],
            [0, 0.001, [['soma', '0.5', 'nap_v', 'm_tau_min']]],
            [0, 0.5, [['soma', '0.5', 'nap_v', 'h_tau_min']]],
            [0, 0.001, [['soma', '0.5', 'kdr_v', 'm_tau_min']]],
            [10, 20, [['soma', '0.5', 'nat_v', 'm_tau_max']]],
            [12, 23, [['soma', '0.5', 'nat_v', 'h_tau_max']]],
            [0, 0.1, [['soma', '0.5', 'nap_v', 'm_tau_max']]],
            [5, 15, [['soma', '0.5', 'nap_v', 'h_tau_max']]],
            [15, 25, [['soma', '0.5', 'kdr_v', 'm_tau_max']]],
            [0.2, 0.8, [['soma', '0.5', 'nat_v', 'm_tau_delta']]],
            [0.7, 1.3, [['soma', '0.5', 'nat_v', 'h_tau_delta']]],
            [0, 0.4, [['soma', '0.5', 'nap_v', 'm_tau_delta']]],
            [0, 0.5, [['soma', '0.5', 'nap_v', 'h_tau_delta']]],
            [0.2, 0.8, [['soma', '0.5', 'kdr_v', 'm_tau_delta']]],
            ]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}
lower_bounds_init, upper_bounds_init, variable_keys_init = get_lowerbound_upperbound_keys(variables_init)
bounds_init = {'lower_bounds': list(lower_bounds_init), 'upper_bounds': list(upper_bounds_init)}

fitter_params = {
                    'name': 'HodgkinHuxleyFitterSeveralData',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'fitfun_names': ['get_v'],
                    'fitnessweights': [1],
                    'model_dir': '../../model/cells/nowacki_model2.json',
                    'mechanism_dir': '../../model/channels/nowacki',
                    'data_dirs': ['../../data/2015_08_06d/vrest-81/PP(4)/0(nA).csv',
                                  '../../data/2015_08_06d/vrest-81/IV/-0.1(nA).csv'],
                    'simulation_params': [
                                          {'celsius': 35, 'onset': 200},
                                          {'celsius': 35, 'onset': 200}
                                          ],
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