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
            [0.3, 2, [['soma', 'cm']]],
            [-95, -70, [['soma', '0.5', 'pas', 'e']]],
            [-30, -10, [['soma', '0.5', 'hcn', 'ehcn']]],

            [0, 0.1, [['soma', '0.5', 'pas', 'g']]],
            [0, 0.1, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 0.1, [['soma', '0.5', 'hcn', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'nat', 'gbar']]],

            [0, 5, [['soma', '0.5', 'kdr', 'o_pow']]],
            [0, 5, [['soma', '0.5', 'hcn', 'o_pow']]],

            [0.0001, 100, [['soma', '0.5', 'kdr', 'a1_0']]],
            [0.0001, 10, [['soma', '0.5', 'kdr', 'a1_1']]],
            [0.0001, 100, [['soma', '0.5', 'kdr', 'b1_0']]],
            [0.0001, 10, [['soma', '0.5', 'kdr', 'b1_1']]],
            [0.0001, 100, [['soma', '0.5', 'hcn', 'a1_0']]],
            [0.0001, 10, [['soma', '0.5', 'hcn', 'a1_1']]],
            [0.0001, 100, [['soma', '0.5', 'hcn', 'b1_0']]],
            [0.0001, 10, [['soma', '0.5', 'hcn', 'b1_1']]],
            [0.0001, 100, [['soma', '0.5', 'nat', 'a1_0']]],
            [0.0001, 10, [['soma', '0.5', 'nat', 'a1_1']]],
            [0.0001, 100, [['soma', '0.5', 'nat', 'b1_0']]],
            [0.0001, 10, [['soma', '0.5', 'nat', 'b1_1']]],
            [0.0001, 100, [['soma', '0.5', 'nat', 'a2_0']]],
            [0.0001, 10, [['soma', '0.5', 'nat', 'a2_1']]],
            [0.0001, 100, [['soma', '0.5', 'nat', 'b2_0']]],
            [0.0001, 10, [['soma', '0.5', 'nat', 'b2_1']]],
            [0.0001, 100, [['soma', '0.5', 'nat', 'a3_0']]],
            [0.0001, 10, [['soma', '0.5', 'nat', 'a3_1']]],
            [0.0001, 100, [['soma', '0.5', 'nat', 'b3_0']]],
            [0.0001, 10, [['soma', '0.5', 'nat', 'b3_1']]],
            ]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}

fitter_params = {
                    'name': 'HodgkinHuxleyFitterSeveralDataSeveralFitfuns',
                    #'name': 'HodgkinHuxleyFitter',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'fitfun_names': [['get_v', 'get_DAP'], ['get_v'], ['get_v', 'get_n_spikes']],
                    'model_dir': '../../model/cells/dapmodel_simpel.json',
                    'mechanism_dir': '../../model/channels/markov',
                    # 'fitnessweights': [1, 1000],
                    # 'data_dir': '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv',
                    # 'simulation_params': {'celsius': 35, 'onset': 200},
                    'fitnessweights': [[100, 1], [1], [0.01, 5]],
                    'data_dirs': [
                                  '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv',
                                  '../../data/2015_08_26b/vrest-75/IV/-0.1(nA).csv',
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
                            bounds['lower_bounds'],
                            bounds['upper_bounds'],
                            optimization_settings_dict['seed'],
                            optimization_settings_dict['n_candidates'])

# choose right candidate
batch_size = sys.argv[3]
optimization_settings_dict['extra_args']['init_candidates'] = init_candidates[int(sys.argv[2])*int(batch_size):
                                                               (int(sys.argv[2])+1)*int(batch_size)]
#optimization_settings_dict['extra_args']['init_candidates'] = init_candidates[0:1]

# start optimization
optimize(optimization_settings_dict, algorithm_settings_dict)