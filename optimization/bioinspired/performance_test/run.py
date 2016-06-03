import inspyred
from random import Random
from time import time
import numpy as np
from optimization.bioinspired.problem import Problem
import json
import os
import pandas as pd

__author__ = 'caro'


def run(problem, method, method_type, method_args, seed, max_generations, individuals_file, normalize):

    # create random number generator with seed
    prng = Random()
    prng.seed(seed)

    # create optimizer
    ea = getattr(getattr(inspyred, method_type), method)(prng)
    if method == 'GA':
        ea.variator = [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.gaussian_mutation]

    # setup and run optimization algorithm
    ea.observer = problem.observer  #inspyred.ec.observers.file_observer
    ea.terminator = inspyred.ec.terminators.generation_termination
    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator,
                          maximize=problem.maximize,
                          bounder=inspyred.ec.Bounder(problem.lower_bound, problem.upper_bound),
                          individuals_file=individuals_file,
                          normalize=normalize,
                          max_generations=max_generations,
                          **method_args)

    best = max(final_pop)
    print('Best Solution: \n{0}'.format(str(best)))

if __name__ == '__main__':

    save_dir = './results/test/'
    if not os.path.exists(save_dir):
                os.makedirs(save_dir)

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
    with open(save_dir+'params.json', 'w') as f:
        json.dump(params, f)

    problem = Problem(params)
    pop_size = 10
    n_trials = 20

    methods = ['DEA', 'SA', 'GA', 'EDA', 'PSO']
    method_types = ['ec', 'ec', 'ec', 'ec', 'swarm']
    #method_args = [{'pop_size': pop_size, 'num_selected': 67, 'tournament_size': 36,
    #               'crossover_rate': 0.57, 'mutation_rate': 0.52, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.21},
    #              {'temperature': 524.25, 'cooling_rate': 0.51, 'mutation_rate': 0.68,
    #               'gaussian_mean': 0.0, 'gaussian_stdev': 0.20},
    #              {'pop_size': pop_size, 'num_selected': 100,
    #               'crossover_rate': 0.44, 'num_crossover_points': 1, 'mutation_rate': 0.53,
    #               'gaussian_mean': 0.0, 'gaussian_stdev': 0.10, 'num_elites': 28},
    #              {'pop_size': pop_size, 'num_selected': 49, 'num_offspring': 100, 'num_elites': 27},
    #              {'pop_size': pop_size, 'inertia': 0.43, 'cognitive_rate': 1.44, 'social_rate': 1.57}
    #              ]  # for errfun_pointtopoint
    method_args = [{'pop_size': pop_size, 'num_selected': 68, 'tournament_size': 49,
                   'crossover_rate': 0.51, 'mutation_rate': 0.60, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.31},
                  {'temperature': 566.14, 'cooling_rate': 0.49, 'mutation_rate': 0.68,
                   'gaussian_mean': 0.0, 'gaussian_stdev': 0.38},
                  {'pop_size': pop_size, 'num_selected': 100,
                   'crossover_rate': 0.46, 'num_crossover_points': 1, 'mutation_rate': 0.48,
                   'gaussian_mean': 0.0, 'gaussian_stdev': 0.21, 'num_elites': 27},
                  {'pop_size': pop_size, 'num_selected': 34, 'num_offspring': 100, 'num_elites': 26},
                  {'pop_size': pop_size, 'inertia': 0.44, 'cognitive_rate': 1.42, 'social_rate': 1.5}
                  ]  # for errfun_featurebased

    max_generations = 100

    for trial in range(n_trials):
        #seed = time()
        #np.savetxt(save_dir+'seed'+str(trial)+'.txt', np.array([seed]))
        #seed = float(np.loadtxt(save_dir+'seed'+str(trial)+'.txt'))
        seed = float(np.loadtxt('./results/errfun_pointtopoint/seed'+str(trial)+'.txt'))
        for i in range(len(methods)):
            max_generations_a = max_generations
            if methods[i] == 'SA':
                max_generations_a = max_generations * pop_size
            if not os.path.exists(save_dir+methods[i]):
                os.makedirs(save_dir+methods[i])
            individuals_file = open(save_dir+methods[i]+'/individuals_file_'+str(trial)+'.csv', 'w')

            # run
            run(problem, methods[i], method_types[i], method_args[i], seed, max_generations_a,
                individuals_file, normalize=True)