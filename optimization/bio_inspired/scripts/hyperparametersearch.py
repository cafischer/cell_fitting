from random import Random
from time import time
import numpy as np
import json
import os
import inspyred

from util import merge_dicts
from optimization.problem import CellFitProblem
from optimization.bio_inspired.optimize import evolve

__author__ = 'caro'


def generate_hyperparameter(arg_bounds, seed, round=False):
    prng = Random()
    prng.seed(seed)
    args = dict()
    for key in arg_bounds.keys():
        if arg_bounds[key][2] == 'float':
            if round:
                args[key] = np.round(prng.uniform(arg_bounds[key][0], arg_bounds[key][1]), 2)
            else:
                args[key] = prng.uniform(arg_bounds[key][0], arg_bounds[key][1])
        elif arg_bounds[key][2] == 'int':
             args[key] = prng.randint(arg_bounds[key][0], arg_bounds[key][1])
        else:
            raise ValueError('Wrong parameter type specification!')
    return args


if __name__ == '__main__':

    save_dir = './results_hyperparametersearch/errfun_featurebased/'

    path_variables = [[['soma', 'mechanisms', 'naf', 'gbar']],
                      [['soma', 'mechanisms', 'ka', 'gbar']]]

    params = {'data_dir': './testdata/modeldata.csv',
              'model_dir': '../../../model/cells/dapmodel.json',
              'mechanism_dir': '../../../model/channels_currentfitting',
              'lower_bound': 0, 'upper_bound': 1,
              'maximize': False,
              'fun_to_fit': 'run_simulation', 'var_to_fit': 'v',
              'path_variables': path_variables,
              'errfun': 'errfun_featurebased'}

    problem = CellFitProblem(params)
    methods = ['DEA', 'SA', 'GA', 'EDA', 'PSO']

    idx = 0  # select method
    pop_size = 100
    n_trials = 1000
    max_generations = 100
    max_evaluations = max_generations
    if methods[idx] == 'SA':
        max_evaluations = max_generations * pop_size

    method_types = ['ec', 'ec', 'ec', 'ec', 'swarm']
    method_args = [{}, {}, {}, {}, {}, {}]
    arg_bounds = [{'pop_size': [pop_size, pop_size, 'int'],
                   'num_selected': [1, pop_size, 'int'], 'tournament_size': [2, pop_size, 'int'],
                   'crossover_rate': [0, 1, 'float'], 'mutation_rate': [0, 1, 'float'],
                   'gaussian_mean': [0, 0, 'float'], 'gaussian_stdev': [0.001, 1, 'float']},
                  {'temperature': [0, 1000, 'float'], 'cooling_rate': [0, 1, 'float'],
                   'mutation_rate': [0, 1, 'float'],
                   'gaussian_mean': [0, 0, 'float'], 'gaussian_stdev': [0.001, 1, 'float']},
                  {'pop_size': [pop_size, pop_size, 'int'],
                   'num_selected': [pop_size, pop_size, 'int'],
                   'crossover_rate': [0, 1, 'float'], 'num_crossover_points':[1, 1, 'int'],
                   'mutation_rate': [0, 1, 'float'],
                   'gaussian_mean': [0, 0, 'float'], 'gaussian_stdev': [0.001, 1, 'float'],
                   'num_elites': [0, pop_size/2, 'int']},
                  {'pop_size': [pop_size, pop_size, 'int'], 'num_selected': [1, pop_size, 'int'],
                   'num_offspring': [pop_size, pop_size, 'int'], 'num_elites': [0, pop_size/2, 'int']},
                  {'pop_size': [pop_size, pop_size, 'int'], 'inertia': [0, 2, 'float'],
                   'cognitive_rate': [0, 3, 'float'], 'social_rate': [0, 3, 'float']}
                  ]
    method = methods[idx]
    method_type = method_types[idx]
    method_arg = method_args[idx]

    if not os.path.exists(save_dir+methods[idx]):
            os.makedirs(save_dir+methods[idx])

    with open(save_dir+'params.json', 'w') as f:
        json.dump(params, f)

    for trial in range(n_trials):
        # create random number generator with seed and save seed
        if os.path.isfile(save_dir+'/seed'+str(trial)+'.txt'):
            seed = float(np.loadtxt(save_dir+'seed_'+str(trial)+'.txt'))
        else:
            seed = time()
            np.savetxt(save_dir+'/seed_'+str(trial)+'.txt', np.array([seed]))
        prng = Random()
        prng.seed(seed)

        # sample parameter from uniform distribution
        seed_sampling = time()
        with open(save_dir+methods[idx]+'/seed_sampling_'+str(trial)+'.npy', 'w') as f:
            np.save(f, seed_sampling)
        method_args[idx] = generate_hyperparameter(arg_bounds[idx], seed_sampling, round=True)
        with open(save_dir+methods[idx]+'/method_args_'+str(trial)+'.json', 'w') as f:
            json.dump(method_args[idx], f)

        # get optimization method
        ea = getattr(getattr(inspyred, method_type), method)(prng)
        if method == 'GA':
            ea.variator = [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.gaussian_mutation]
        ea.observer = problem.observer
        ea.terminator = inspyred.ec.terminators.generation_termination

        # open individuals_file and add it to args
        if not os.path.exists(save_dir+'/'+method):
                os.makedirs(save_dir+'/'+method)
        individuals_file = open(save_dir+'/'+method+'/individuals_file_'+str(trial)+'.csv', 'w')
        args = merge_dicts(method_arg, {'individuals_file': individuals_file})

        # run optimization
        evolve(ea, problem, **args)

        individuals_file.close()