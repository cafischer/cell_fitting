from time import time

from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
from cell_fitting.optimization.scripts import optimize

save_dir = '../../results/test_dtw/'

variables = [
                [0.5, 2.5, [['soma', 'cm']]],
                [0, 1.0, [['soma', '0.5', 'na_hh', 'gnabar']]],
                [0, 1.0, [['soma', '0.5', 'k_hh', 'gkbar']]],
                [0, 1.0, [['soma', '0.5', 'pas', 'g']]],
                [20, 100, [['soma', 'ena']]],
                [-100, -20, [['soma', 'ek']]],
                [-100, -20, [['soma', '0.5', 'pas', 'e']]],
                [0, 10, [['soma', '0.5', 'na_hh', 'alpha_m_f']]],
                [0, 80, [['soma', '0.5', 'na_hh', 'alpha_m_v']]],
                [1, 90, [['soma', '0.5', 'na_hh', 'alpha_m_k']]],
                [0, 10, [['soma', '0.5', 'na_hh', 'beta_m_f']]],
                [0, 80, [['soma', '0.5', 'na_hh', 'beta_m_v']]],
                [1, 90, [['soma', '0.5', 'na_hh', 'beta_m_k']]],
                [0, 10, [['soma', '0.5', 'na_hh', 'alpha_h_f']]],
                [0, 80, [['soma', '0.5', 'na_hh', 'alpha_h_v']]],
                [1, 90, [['soma', '0.5', 'na_hh', 'alpha_h_k']]],
                [0, 80, [['soma', '0.5', 'na_hh', 'beta_h_v']]],
                [1, 90, [['soma', '0.5', 'na_hh', 'beta_h_k']]],
                [0, 10, [['soma', '0.5', 'k_hh', 'alpha_n_f']]],
                [0, 80, [['soma', '0.5', 'k_hh', 'alpha_n_v']]],
                [1, 90, [['soma', '0.5', 'k_hh', 'alpha_n_k']]],
                [0, 10, [['soma', '0.5', 'k_hh', 'beta_n_f']]],
                [0, 80, [['soma', '0.5', 'k_hh', 'beta_n_v']]],
                [1, 90, [['soma', '0.5', 'k_hh', 'beta_n_k']]]
                ]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
bounds = {'lower_bounds': list(lower_bounds), 'upper_bounds': list(upper_bounds)}

fitter_params = {
                    'name': 'HodgkinHuxleyFitter',
                    'variable_keys': variable_keys,
                    'errfun_name': 'meanabs_error',
                    'fitfun_names': ['get_v'],
                    'fitnessweights': [1],
                    'model_dir': '../../model/cells/hhCell.json',
                    'mechanism_dir': '../../model/channels/hodgkinhuxley',
                    'data_dir': '../../data/toymodels/hhCell/ramp.csv',
                    'simulation_params': {'celsius': 6.3},
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
#algorithm_name = 'DEA'
#algorithm_params = {'num_selected': 335, 'tournament_size': 180, 'crossover_rate': 0.57,
#                    'mutation_rate': 0.52, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.21}
#algorithm_name = 'SA'
#algorithm_params = {'temperature': 524.25, 'cooling_rate': 0.51, 'mutation_rate': 0.68,
#                    'gaussian_mean': 0.0, 'gaussian_stdev': 0.20}
#algorithm_name = 'PSO'
#algorithm_params = {'inertia': 0.43, 'cognitive_rate': 1.44, 'social_rate': 1.57}
#algorithm_name = 'rmsprop'
#algorithm_params = {'step_rate': 0.1, 'decay': 0.9, 'momentum': 0.2, 'step_adapt': False,
#                    'step_rate_min': 0, 'step_rate_max': np.inf, 'args': None}
#algorithm_name = 'adadelta'
#algorithm_params = {'step_rate': 1, 'decay': 0.9, 'momentum': 0, 'offset': 0.0001,
#                   'args': None}
#algorithm_name = 'adam'
#algorithm_params = {'step_rate': 0.0002, 'decay': None, 'decay_mom1': 0.1, 'decay_mom2': 0.001, 'momentum': 0, 'offset': 1e-08,
#                   'args': None}
#algorithm_name = 'Nelder-Mead'
#algorithm_params = {}
#algorithm_name = 'Random'
#algorithm_params = {}

optimize(optimization_settings_dict, algorithm_settings_dict)