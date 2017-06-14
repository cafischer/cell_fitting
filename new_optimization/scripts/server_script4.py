import sys
sys.path.append("../../")
from optimization.helpers import get_lowerbound_upperbound_keys
from time import time
from new_optimization.scripts import optimize
import sys
from new_optimization import generate_initial_candidates
import os


# parameters
save_dir = sys.argv[1]

variables = [
            [0.1, 1.5, [['soma', 'cm']]],
            [-100, -50, [['soma', '0.5', 'pas', 'e']]],

            [0, 0.001, [['soma', '0.5', 'pas', 'g']]],
            [0, 0.5, [['soma', '0.5', 'nap', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'nat2', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'km', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'cah', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'hcn', 'gbar']]],

            [-90, 0, [['soma', '0.5', 'nap', 'm_vh']]],
            [-90, 0, [['soma', '0.5', 'nat2', 'm_vh']]],
            [-90, 0, [['soma', '0.5', 'nat2', 'h_vh']]],
            [-90, 0, [['soma', '0.5', 'kdr', 'm_vh']]],
            [-90, 0, [['soma', '0.5', 'kdr', 'h_vh']]],
            [-90, 0, [['soma', '0.5', 'km', 'm_vh']]],
            [-90, 0, [['soma', '0.5', 'cah', 'm_vh']]],
            [-90, 0, [['soma', '0.5', 'cah', 'h_vh']]],
            [-90, 0, [['soma', '0.5', 'hcn', 'h_vh']]],

            [1, 30, [['soma', '0.5', 'nap', 'm_vs']]],
            [1, 30, [['soma', '0.5', 'nat2', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nat2', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'kdr', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'kdr', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'km', 'm_vs']]],
            [1, 30, [['soma', '0.5', 'cah', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'cah', 'h_vs']]],
            [-30, -1, [['soma', '0.5', 'hcn', 'h_vs']]],

            [0, 50, [['soma', '0.5', 'nat2', 'h_tau_min']]],
            [0.001, 50, [['soma', '0.5', 'nat2', 'h_tau_max']]],
            [0, 5, [['soma', '0.5', 'nat2', 'h_tau_delta']]],
            [0, 20, [['soma', '0.5', 'kdr', 'mtau']]],
            [0, 100, [['soma', '0.5', 'km', 'mtau']]],
            [100, 1000, [['soma', '0.5', 'kdr', 'htau']]],
            [0, 20, [['soma', '0.5', 'cah', 'mtau']]],
            [0, 500, [['soma', '0.5', 'cah', 'htau']]],
            [0, 1000, [['soma', '0.5', 'hcn', 'htau']]],
            ]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}

fitter_params = {
                    'name': 'HodgkinHuxleyFitter',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'fitfun_names': ['get_v'],
                    'fitnessweights': [1],
                    'model_dir': '../../model/cells/nowacki_model.json',
                    'mechanism_dir': '../../model/channels/nowacki',
                    'data_dir': '../../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(3)/0(nA).csv',
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
                            optimization_settings_dict['bounds']['lower_bounds'],
                            optimization_settings_dict['bounds']['upper_bounds'],
                            optimization_settings_dict['seed'],
                            optimization_settings_dict['n_candidates'])

# choose right candidate
batch_size = sys.argv[3]
optimization_settings_dict['extra_args']['init_candidates'] = init_candidates[int(sys.argv[2])*int(batch_size):
                                                               (int(sys.argv[2])+1)*int(batch_size)]

# start optimization
optimize(optimization_settings_dict, algorithm_settings_dict)