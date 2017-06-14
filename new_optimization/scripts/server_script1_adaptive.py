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
            [0.6, 2, [['soma', 'cm']]],
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
            [1.0, 1.4, [['soma', 'cm']]],
            [-75, -60, [['soma', '0.5', 'pas', 'e']]],

            [0.001, 0.005, [['soma', '0.5', 'pas', 'g']]],
            [0.05, 0.11, [['soma', '0.5', 'nat', 'gbar']]],
            [0.01, 0.05, [['soma', '0.5', 'kdr', 'gbar']]],
            [0.1, 0.4, [['soma', '0.5', 'nap', 'gbar']]],

            [-44, -34, [['soma', '0.5', 'nat', 'm_vh']]],
            [-74, -61, [['soma', '0.5', 'nat', 'h_vh']]],
            [-59, -46, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-23, -13, [['soma', '0.5', 'nap', 'm_vh']]],
            [-62, -52, [['soma', '0.5', 'nap', 'h_vh']]],

            [11, 17, [['soma', '0.5', 'nat', 'm_vs']]],
            [-25, -19, [['soma', '0.5', 'nat', 'h_vs']]],
            [13, 19, [['soma', '0.5', 'kdr', 'n_vs']]],
            [16, 22, [['soma', '0.5', 'nap', 'm_vs']]],
            [-16, -10, [['soma', '0.5', 'nap', 'h_vs']]],

            [0, 3, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0, 4, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0, 0.001, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 0.001, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0, 3, [['soma', '0.5', 'nap', 'h_tau_min']]],

            [10, 20, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [12, 22, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [15, 25, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0.001, 3, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [5, 15, [['soma', '0.5', 'nap', 'h_tau_max']]],

            [0.2, 0.8, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0.6, 1.2, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0.2, 0.8, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0.01, 0.4, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0.01, 0.4, [['soma', '0.5', 'nap', 'h_tau_delta']]],
            ]


lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}
lower_bounds_init, upper_bounds_init, variable_keys_init = get_lowerbound_upperbound_keys(variables_init)
bounds_init = {'lower_bounds': list(lower_bounds_init), 'upper_bounds': list(upper_bounds_init)}

# discontinuities for IV
dt = 0.05
start_step = int(round(250 / dt))
end_step = int(round(750 / dt))
discontinuities_IV = [start_step, end_step]

fitter_params = {
                    'name': 'HodgkinHuxleyFitterSeveralDataAdaptive',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'fitfun_names': ['get_v'],
                    'fitnessweights': [1],
                    'model_dir': '../../model/cells/dapmodel_simpel.json',
                    'mechanism_dir': '../../model/channels/vavoulis',
                    'data_dirs': ['../../data/2015_08_26b/vrest-60/rampIV/3.0(nA).csv',
                                  '../../data/2015_08_26b/vrest-60/IV/-0.1(nA).csv'],
                    'simulation_params': [
                                          {'celsius': 35, 'onset': 200, 'atol': 1e-8, 'continuous': True,
                                          'discontinuities': None, 'interpolate': True},
                                          {'celsius': 35, 'onset': 200, 'atol': 1e-8, 'continuous': True,
                                           'discontinuities': discontinuities_IV, 'interpolate': True}
                                          ],
                    'args': {}
                }

optimization_settings_dict = {
    'maximize': False,
    'n_candidates': 10000,
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