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
            [-30, -10, [['soma', '0.5', 'hcn_slow', 'ehcn']]],

            [0, 0.5, [['soma', '0.5', 'pas', 'g']]],
            [0, 0.5, [['soma', '0.5', 'nat', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 1.0, [['soma', '0.5', 'nap', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'hcn_slow', 'gbar']]],

            [-100, 0, [['soma', '0.5', 'nat', 'm_vh']]],
            [-100, 0, [['soma', '0.5', 'nat', 'h_vh']]],
            [-100, 0, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-100, 0, [['soma', '0.5', 'nap', 'm_vh']]],
            [-100, 0, [['soma', '0.5', 'nap', 'h_vh']]],
            [-100, 0, [['soma', '0.5', 'hcn_slow', 'n_vh']]],

            [1, 30, [['soma', '0.5', 'nat', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nat', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'kdr', 'n_vs']]],
            [1, 30, [['soma', '0.5', 'nap', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nap', 'h_vs']]],
            [-30, -1, [['soma', '0.5', 'hcn_slow', 'n_vs']]],

            [0, 50, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0, 50, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0, 50, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 50, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0, 50, [['soma', '0.5', 'nap', 'h_tau_min']]],
            [0, 50, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]],

            [0, 100, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [0, 100, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [0, 100, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0, 100, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [0, 100, [['soma', '0.5', 'nap', 'h_tau_max']]],
            [0, 500, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]],

            [0, 10, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0, 10, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0, 10, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 10, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0, 10, [['soma', '0.5', 'nap', 'h_tau_delta']]],
            [0, 10, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
            ]

variables_init = [
            [0.65, 0.72, [['soma', 'cm']]],
            [-90, -86, [['soma', '0.5', 'pas', 'e']]],
            [-30, -26, [['soma', '0.5', 'hcn_slow', 'ehcn']]],

            [0.0007, 0.001, [['soma', '0.5', 'pas', 'g']]],
            [0.008, 0.02, [['soma', '0.5', 'nat', 'gbar']]],
            [0.0025, 0.006, [['soma', '0.5', 'kdr', 'gbar']]],
            [0.07, 0.12, [['soma', '0.5', 'nap', 'gbar']]],
            [0.0001, 0.0004, [['soma', '0.5', 'hcn_slow', 'gbar']]],

            [-56, -54, [['soma', '0.5', 'nat', 'm_vh']]],
            [-80, -78, [['soma', '0.5', 'nat', 'h_vh']]],
            [-69, -67, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-36, -32, [['soma', '0.5', 'nap', 'm_vh']]],
            [-74, -70, [['soma', '0.5', 'nap', 'h_vh']]],
            [-85, -83, [['soma', '0.5', 'hcn_slow', 'n_vh']]],

            [15, 17, [['soma', '0.5', 'nat', 'm_vs']]],
            [-22, -20, [['soma', '0.5', 'nat', 'h_vs']]],
            [18, 20, [['soma', '0.5', 'kdr', 'n_vs']]],
            [14, 18, [['soma', '0.5', 'nap', 'm_vs']]],
            [-15, -11, [['soma', '0.5', 'nap', 'h_vs']]],
            [-21, -19, [['soma', '0.5', 'hcn_slow', 'n_vs']]],

            [0.01, 0.03, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0.4, 0.6, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0.5, 0.8, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 0.01, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0, 0.1, [['soma', '0.5', 'nap', 'h_tau_min']]],
            [3, 5, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]],

            [17, 19, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [17, 19, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [20, 22, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0.2, 0.4, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [7, 9, [['soma', '0.5', 'nap', 'h_tau_max']]],
            [90, 150, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]],

            [0.3, 0.5, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0.6, 0.8, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0.6, 0.8, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0.1, 0.3, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0.3, 0.4, [['soma', '0.5', 'nap', 'h_tau_delta']]],
            [0.2, 0.4, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
            ]


lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}
lower_bounds_init, upper_bounds_init, variable_keys_init = get_lowerbound_upperbound_keys(variables_init)
bounds_init = {'lower_bounds': list(lower_bounds_init), 'upper_bounds': list(upper_bounds_init)}

# discontinuities for plot_IV
dt = 0.05
start_step = int(round(250 / dt))
end_step = int(round(750 / dt))
discontinuities_IV = [start_step, end_step]

fitter_params = {
                    'name': 'HodgkinHuxleyFitterSeveralDataAdaptive',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'fitfun_names': ['get_v'],
                    'fitnessweights': [150, 1],
                    'model_dir': '../../model/cells/dapmodel_simpel.json',
                    'mechanism_dir': '../../model/channels/vavoulis',
                    'data_dirs': [
                                  '../../data/2015_08_26b/vrest-75/simulate_rampIV/3.0(nA).csv',
                                  '../../data/2015_08_26b/vrest-75/plot_IV/0.3(nA).csv'
                                  ],
                    'simulation_params': [
                                         {'celsius': 35, 'onset': 200, 'atol': 1e-5, 'continuous': True,
                                         'discontinuities': None, 'interpolate': True},
                                         {'celsius': 35, 'onset': 200, 'atol': 1e-5, 'continuous': True,
                                         'discontinuities': discontinuities_IV, 'interpolate': True}
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