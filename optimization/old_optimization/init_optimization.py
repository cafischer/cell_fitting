from random import shuffle

from nrn_wrapper import load_mechanism_dir

from optimization import save_problem_specification
from optimization.bio_inspired.generators import *
from optimization.old_optimization.evaluation import Evaluator

__author__ = 'caro'


def check_if_optimal_candidate_in_range_variables(mechanism_dir, model_dir, variables_all):

    evaluator = Evaluator([], [save_dir], 0, [])
    load_mechanism_dir(mechanism_dir)
    optimal_candidate = evaluator.get_optimal_candidate(model_dir, variables_all)

    for i in range(len(variables_all)):
        assert variables_all[i][0] <= optimal_candidate[i] <= variables_all[i][1]


# parameter
save_dir = '../../results/algorithms_on_hhcell/10param/'
n_trials = 100
pop_size = 250
max_iterations = 250
n_vars = 10  # number of variables to adapt in the evolution

# specify problem
variables_all = [
                [0, 1.5, [['soma', '0.5', 'na_hh', 'gnabar']]],
                [0, 1.5, [['soma', '0.5', 'k_hh', 'gkbar']]],
                [0, 1.5, [['soma', '0.5', 'pas', 'g']]],
                [30, 80, [['soma', '0.5', 'ena']]],
                [-100, -60, [['soma', '0.5', 'ek']]],
                [-75, -35, [['soma', '0.5', 'pas', 'e']]],
                [0.5, 2.0, [['soma', 'cm']]],
                [0, 5, [['soma', '0.5', 'na_hh', 'alpha_m_f']]],
                [30, 70, [['soma', '0.5', 'na_hh', 'alpha_m_v']]],
                [0, 50, [['soma', '0.5', 'na_hh', 'alpha_m_k']]],
                [0, 5, [['soma', '0.5', 'na_hh', 'beta_m_f']]],
                [30, 70, [['soma', '0.5', 'na_hh', 'beta_m_v']]],
                [0, 50, [['soma', '0.5', 'na_hh', 'beta_m_k']]],
                [0, 5, [['soma', '0.5', 'na_hh', 'alpha_h_f']]],
                [30, 70, [['soma', '0.5', 'na_hh', 'alpha_h_v']]],
                [0, 50, [['soma', '0.5', 'na_hh', 'alpha_h_k']]],
                [30, 70, [['soma', '0.5', 'na_hh', 'beta_h_v']]],
                [0, 50, [['soma', '0.5', 'na_hh', 'beta_h_k']]],
                [0, 5, [['soma', '0.5', 'k_hh', 'alpha_n_f']]],
                [30, 70, [['soma', '0.5', 'k_hh', 'alpha_n_v']]],
                [0, 50, [['soma', '0.5', 'k_hh', 'alpha_n_k']]],
                [0, 5, [['soma', '0.5', 'k_hh', 'beta_n_f']]],
                [30, 70, [['soma', '0.5', 'k_hh', 'beta_n_v']]],
                [50, 100, [['soma', '0.5', 'k_hh', 'beta_n_k']]],
                ]

problem_dicts = list()
for trial in range(n_trials):
    indices = range(len(variables_all))
    shuffle(indices)
    mask = indices[:n_vars]
    variables = [variables_all[i] for i in mask]
    problem_dict = {
              'name': 'FromInitPopCellFitProblem',
              'maximize': False,
              'normalize': True,
              'model_dir': '../../model/cells/hhCell.json',
              'mechanism_dir': '../../model/channels/hodgkinhuxley',
              'variables': variables,
              'data_dir': '../../data/toymodels/hhCell/ramp.csv',
              'get_var_to_fit': 'get_v',
              'fitnessweights': [1.0],
              'errfun': 'rms',
              'insert_mechanisms': True,
              'init_pop': [],
              'simulation_params': {'sec': ('soma', None), 'celsius': 6.3}
             }
    problem_dicts.append(problem_dict)

generator = get_random_numbers_in_bounds  # save unnormalized version!

#check_if_optimal_candidate_in_range_variables(problem_dict['mechanism_dir'], problem_dict['model_dir'], variables_all)
save_problem_specification(save_dir, n_trials, pop_size, max_iterations, problem_dicts, variables_all, generator)

