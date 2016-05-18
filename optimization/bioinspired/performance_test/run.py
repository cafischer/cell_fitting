import inspyred
from random import Random
from time import time
import numpy as np
from optimization.bioinspired.problem import Problem
import json
import os

__author__ = 'caro'


def run(problem, method, method_type, method_args, seed, max_generations, individuals_file, statistics_file):

    # create random number generator with seed
    prng = Random()
    prng.seed(seed)

    # create optimizer
    ea = getattr(getattr(inspyred, method_type), method)(prng)
    if method == 'GA':
        ea.variator = [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.gaussian_mutation]

    # setup and run optimization algorithm
    ea.observer = inspyred.ec.observers.file_observer
    ea.terminator = inspyred.ec.terminators.generation_termination
    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator,
                          maximize=problem.maximize,
                          bounder=inspyred.ec.Bounder(problem.lower_bound, problem.upper_bound),
                          statistics_file=statistics_file,
                          individuals_file=individuals_file,
                          max_generations=max_generations,
                          **method_args)

    best = max(final_pop)
    print('Best Solution: \n{0}'.format(str(best)))

if __name__ == '__main__':

    path_variables = [[['soma', 'mechanisms', 'naf', 'gbar']],
                      [['soma', 'mechanisms', 'ka', 'gbar']]]

    params = {'data_dir': './testdata/modeldata.csv',
              'model_dir': '../../../model/cells/dapmodel.json',
              'mechanism_dir': '../../../model/channels_currentfitting',
              'lower_bound': 0, 'upper_bound': 1,
              'maximize': False,
              'fun_to_fit': 'run_simulation', 'var_to_fit': 'v',
              'path_variables': path_variables,
              'errfun': 'errfun_pointtopoint'}
    with open('./results/params.json', 'w') as f:
        json.dump(params, f)

    problem = Problem(params)
    pop_size = 100
    n_trials = 100

    methods = ['DEA', 'SA', 'GA', 'EDA', 'PSO']
    method_types = ['ec', 'ec', 'ec', 'ec', 'swarm']
    method_args = [
        {'tournament_size': 7, 'crossover_rate': 0.91, 'gaussian_mean': 0.0, 'num_selected': 86,
         'gaussian_stdev': 0.49, 'mutation_rate': 0.16, 'pop_size': 100}
,
        {},
        {'pop_size': pop_size, 'num_elites': 1},
        {'pop_size': pop_size, 'num_elites': 1, 'num_selected': int(pop_size*0.5), 'num_offspring': int(pop_size*0.75)},
        {'pop_size': pop_size, 'neighborhood_size': 5}]
    max_evaluations = 10000

    for trial in range(n_trials):
        seed = time()
        with open('./results/seed'+str(trial)+'.npy', 'w') as f:
            np.save(f, seed)
        for i in range(len(methods)):
            if not os.path.exists('./results/'+methods[i]):
                os.makedirs('./results/'+methods[i])
            individuals_file = open('./results/'+methods[i]+'/individuals_file_'+str(trial)+'.csv', 'w')
            statistics_file = open('./results/'+methods[i]+'/statistics_file_'+str(trial)+'.csv', 'w')

            # run
            run(problem, methods[i], method_types[i], method_args[i], seed, max_evaluations,
                individuals_file, statistics_file)