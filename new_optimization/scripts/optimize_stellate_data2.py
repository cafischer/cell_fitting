from optimization.helpers import get_lowerbound_upperbound_keys
from time import time
from new_optimization.scripts import optimize

save_dir = '../../results/new_optimization/2015_08_06d/31_03_17_modified_nat/'

variables = [
            [0.5, 1.5, [['soma', 'cm']]],
            [-100, -50, [['soma', '0.5', 'pas', 'e']]],
            #[-30, -10, [['soma', 'ehcn']]],
            #[50, 70, [['soma', 'ena']]],
            #[-120, -90, [['soma', 'ek']]],

            [0, 0.001, [['soma', '0.5', 'pas', 'g']]],
            [0, 1, [['soma', '0.5', 'nat_modified', 'gbar']]],
            [0, 1, [['soma', '0.5', 'nap', 'gbar']]],
            [0, 1, [['soma', '0.5', 'ka', 'gbar']]],
            [0, 1, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 1, [['soma', '0.5', 'hcn_fast', 'gbar']]],

            [-100, -10, [['soma', '0.5', 'nat_modified', 'm_vh']]],
            [-100, -10, [['soma', '0.5', 'nat_modified', 'r_vh']]],
            [-100, -10, [['soma', '0.5', 'nat_modified', 'h_vh']]],
            [-100, -10, [['soma', '0.5', 'nap', 'm_vh']]],
            [-100, -10, [['soma', '0.5', 'nap', 'h_vh']]],
            [-100, -10, [['soma', '0.5', 'ka', 'n_vh']]],
            [-100, -10, [['soma', '0.5', 'ka', 'l_vh']]],
            [-100, -10, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-100, -10, [['soma', '0.5', 'hcn_fast', 'n_vh']]],

            [1, 30, [['soma', '0.5', 'nat_modified', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nat_modified', 'r_vs']]],
            [-30, -1, [['soma', '0.5', 'nat_modified', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'nap', 'm_vs']]],
            [-30, -1, [['soma', '0.5', 'nap', 'h_vs']]],
            [1, 30, [['soma', '0.5', 'ka', 'n_vs']]],
            [-30, -1, [['soma', '0.5', 'ka', 'l_vs']]],
            [1, 30, [['soma', '0.5', 'kdr', 'n_vs']]],
            [1, 30, [['soma', '0.5', 'hcn_fast', 'n_vs']]],

            [0, 1, [['soma', '0.5', 'nat_modified', 'm_tau_min']]],
            [0.01, 5, [['soma', '0.5', 'nat_modified', 'r_tau_min']]],
            [0.01, 5, [['soma', '0.5', 'nat_modified', 'h_tau_min']]],
            [0, 1, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0.01, 5, [['soma', '0.5', 'nap', 'h_tau_min']]],
            [0, 1, [['soma', '0.5', 'ka', 'n_tau_min']]],
            [0.01, 5, [['soma', '0.5', 'ka', 'l_tau_min']]],
            [0, 1, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 1, [['soma', '0.5', 'hcn_fast', 'n_tau_min']]],

            [0.001, 30, [['soma', '0.5', 'nat_modified', 'm_tau_max']]],
            [0.001, 100, [['soma', '0.5', 'nat_modified', 'r_tau_max']]],
            [0.001, 100, [['soma', '0.5', 'nat_modified', 'h_tau_max']]],
            [0.001, 30, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [0.001, 100, [['soma', '0.5', 'nap', 'h_tau_max']]],
            [0.001, 30, [['soma', '0.5', 'ka', 'n_tau_max']]],
            [0.001, 100, [['soma', '0.5', 'ka', 'l_tau_max']]],
            [0.001, 30, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [0.001, 30, [['soma', '0.5', 'hcn_fast', 'n_tau_max']]],

            [0, 2, [['soma', '0.5', 'nat_modified', 'm_tau_delta']]],
            [0, 2, [['soma', '0.5', 'nat_modified', 'r_tau_delta']]],
            [0, 2, [['soma', '0.5', 'nat_modified', 'h_tau_delta']]],
            [0, 2, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0, 2, [['soma', '0.5', 'nap', 'h_tau_delta']]],
            [0, 2, [['soma', '0.5', 'ka', 'n_tau_delta']]],
            [0, 2, [['soma', '0.5', 'ka', 'l_tau_delta']]],
            [0, 2, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 2, [['soma', '0.5', 'hcn_fast', 'n_tau_delta']]],
]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}

fitter_params = {
                    'name': 'HodgkinHuxleyFitterSeveralData',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'fitfun_names': ['get_v'],
                    'fitnessweights': [1],
                    'model_dir': '../../model/cells/dapmodel_simpel.json',
                    'mechanism_dir': '../../model/channels/vavoulis',
                    'data_dirs': [
                                '../../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(3)/0(nA).csv',
                                '../../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(21)/0(nA).csv'
                                ],
                    'simulation_params': {'celsius': 35, 'onset': 300},
                    'args': {}
                }

optimization_settings_dict = {
    'maximize': False,
    'n_candidates': 1,
    'stop_criterion': ['generation_termination', 2],
    'seed': time(),
    'generator': 'get_random_numbers_in_bounds',
    'bounds': bounds,
    'fitter_params': fitter_params,
    'extra_args': {'multiprocessing': True}
}

algorithm_settings_dict = {
    'algorithm_name': 'L-BFGS-B',
    'algorithm_params': {},
    'optimization_params': {},
    'normalize': False,
    'save_dir': save_dir
}

optimize(optimization_settings_dict, algorithm_settings_dict)