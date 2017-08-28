import os
import numpy as np
from time import time
import json

from optimization import load_problem_specification, problem
from optimization.scipy_based.optimize import optimize_scipy_based
from optimization.bio_inspired.optimize import optimize_bio_inspired, optimize_SA
from optimization.random_draw.optimize import optimize_random
from nrn_wrapper import load_mechanism_dir


__author__ = 'caro'


def run_optimization(save_dir, normalize, return_init_pop, method, method_type, method_args, optimization_type):

    # load problem specification
    n_trials, pop_size, max_iterations, problem_dicts, seeds, init_pops = load_problem_specification(save_dir,
                                                                                                     normalize,
                                                                                                     return_init_pop)

    # load all different mechanisms beforehand
    mechanism_dirs = set([p['mechanism_dir'] for p in problem_dicts])
    for m_dir in mechanism_dirs:
        load_mechanism_dir(m_dir)

    for trial in range(n_trials):

        # make directory for each method and trial
        save_dir_trial = save_dir + '/' + method + '/trial' + str(trial) + '/'
        if not os.path.exists(save_dir_trial):
            os.makedirs(save_dir_trial)

        # save additional method arguments
        with open(save_dir_trial + 'method_args.json', 'w') as f:
            json.dump(method_args, f, indent=4)

        # create problem
        problem_dicts[trial]['mechanism_dir'] = None
        problem = getattr(problem, problem_dicts[trial]['name'])(**problem_dicts[trial])

        # open individuals file
        individuals_file = open(save_dir_trial+'/individuals_file.csv', 'w')

        # measure runtime
        start_time = time()

        # run optimization
        if optimization_type == 'scipy_based':
            optimize_scipy_based(pop_size, max_iterations, method, method_type, method_args, problem, individuals_file)
        elif optimization_type == 'bio_inspired':
            method_args['pop_size'] = pop_size
            method_args['max_generations'] = max_iterations
            seed = float(np.loadtxt(save_dir + 'specification/trial'+str(trial)+'/seed.txt'))
            optimize_bio_inspired(method, method_type, method_args, problem, individuals_file, seed)
        elif optimization_type == 'SA':
            method_args['pop_size'] = 1
            method_args['max_generations'] = max_iterations
            seed = float(np.loadtxt(save_dir + 'specification/trial'+str(trial)+'/seed.txt'))
            optimize_SA(save_dir_trial, pop_size, method, method_type, method_args, problem, individuals_file, seed)
        elif optimization_type == 'random':
            seed = float(np.loadtxt(save_dir + 'specification/trial' + str(trial) + '/seed.txt'))
            optimize_random(pop_size, max_iterations, init_pops[trial], problem, seed, individuals_file)

        # print runtime
        end_time = time()
        print 'Runtime: ' + str(end_time - start_time) + ' sec'

        # close individuals file
        individuals_file.close()



# ---------------------------------------------------------------------------------------------------------------------


# parameter
save_dir = '../../results/algorithms_on_hhcell/6param/'
return_init_pop = True


method = 'Nelder-Mead'
method_type = 'simplex'
method_args = {}
optimization_type = 'scipy_based'
normalize = False
"""
method = 'L-BFGS-B'
method_type = 'gradient_based'
method_args = {'bounds': None}  # specify bound later
optimization_type = 'scipy_based'
normalize = False

method = 'SA'
method_type = 'ec'
method_args = {'temperature': 524.25, 'cooling_rate': 0.51, 'mutation_rate': 0.68,
               'gaussian_mean': 0.0, 'gaussian_stdev': 0.20}
optimization_type = 'SA'
normalize = True

method = 'PSO'
method_type = 'swarm'
method_args = {'inertia': 0.43, 'cognitive_rate': 1.44, 'social_rate': 1.57}
optimization_type = 'bio_inspired'
normalize = True

method = 'DEA'
method_type = 'ec'
method_args = {'num_selected': 335, 'tournament_size': 180, 'crossover_rate': 0.57,
               'mutation_rate': 0.52, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.21}
               # TODO: change for pop_size of 1000: 'num_selected': 670, 'tournament_size': 360,
optimization_type = 'bio_inspired'
normalize = True

method = 'random'
method_type = 'random'
method_args = {}
optimization_type = 'random'
normalize = False
"""


run_optimization(save_dir, normalize, return_init_pop, method, method_type, method_args, optimization_type)
