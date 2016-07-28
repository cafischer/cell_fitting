from random import Random
import json
import os
from time import time
import inspyred
import numpy as np

from nrn_wrapper import Cell
from optimization.bio_inspired.problems import *
from optimization.bio_inspired import problems
from utilities import merge_dicts

__author__ = 'caro'


def search_parameters(algorithm, problem, **args):

    # measure runtime
    starttime = time()

    # setup and run optimization algorithm
    final_pop = algorithm.evolve(generator=problem.generator,
                                 evaluator=problem.evaluator,
                                 maximize=problem.maximize,
                                 bounder=problem.bounder,
                                 **args)

    endtime = time()

    if problem.normalize:
        print 'Best Solution: ' + str(max(unnorm_population(final_pop, problem.lower_bound, problem.upper_bound)))
    else:
        print 'Best Solution: ' + str(max(final_pop))
    print 'Runtime: ' + str(endtime-starttime)


def optimize(save_dir, n_trials, params, method, method_type, method_args, problem='CellFitProblem'):

    # create Problem
    problem = getattr(problems, problem)(**params)

    # save all information
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir+'/problem.json', 'w') as f:
        json.dump(params, f, indent=4)
    with open(save_dir+'/cell.json', 'w') as f:
        json.dump(Cell.from_modeldir(params['model_dir']).get_dict(), f, indent=4)

    # run optimization method on problem
    for trial in range(n_trials):

        # create random number generator with seed and save seed
        if os.path.isfile(save_dir+'/seed'+str(trial)+'.txt'):
            seed = float(np.loadtxt(save_dir+'seed_'+str(trial)+'.txt'))
        else:
            seed = time()
            np.savetxt(save_dir+'/seed_'+str(trial)+'.txt', np.array([seed]))
        prng = Random()
        prng.seed(seed)

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
        args = merge_dicts(method_args, {'individuals_file': individuals_file})

        # run optimization
        search_parameters(ea, problem, **args)

        individuals_file.close()