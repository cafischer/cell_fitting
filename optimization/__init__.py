import json
import os
import numpy as np
from time import time
from random import Random
from neuron import h

from nrn_wrapper import Cell
from optimization import problems
from optimization.problems import complete_mechanismdir
from optimization.problems import norm_candidate, get_variable_information

__author__ = 'caro'


def save_problem_specification(save_dir, n_trials, pop_size, max_iterations, problem_dicts, generator=None):
    save_dir += 'specification/'

    # create save directory
    if os.path.exists(save_dir):
        print 'Saving directory already exists!'
        return False
    os.makedirs(save_dir)

    # save number of trials, population size, maximal number of iterations
    np.savetxt(save_dir+'/n_trials.txt', np.array([n_trials]))
    np.savetxt(save_dir+'/pop_size.txt', np.array([pop_size]))
    np.savetxt(save_dir+'/max_iterations.txt', np.array([max_iterations]))

    # load all different mechanism directories (needed for saving cell)
    mechanism_dirs = set([p['mechanism_dir'] for p in problem_dicts])
    for m_dir in mechanism_dirs:
        h.nrn_load_dll(complete_mechanismdir(m_dir))

    for trial in range(n_trials):
        # create directory for each trial
        save_dir_trial = save_dir + 'trial'+str(trial)+'/'
        os.makedirs(save_dir_trial)

        # save seeds
        seed = time()
        np.savetxt(save_dir_trial+'seed.txt', np.array([seed]))

        # save problem dict
        with open(save_dir_trial+'problem.json', 'w') as f:
            json.dump(problem_dicts[trial], f, indent=4)

        # save cell
        with open(save_dir_trial+'cell.json', 'w') as f:
            json.dump(Cell.from_modeldir(problem_dicts[trial]['model_dir']).get_dict(), f, indent=4)

        # save initial population
        if generator is not None:
            random = Random()
            random.seed(seed)
            initial_pop = list()
            lower_bound, upper_bound, path_variables = get_variable_information(problem_dicts[trial]['variables'])
            for i in range(pop_size):
                initial_pop.append(generator(random, len(path_variables), lower_bound, upper_bound))
            with open(save_dir_trial+'initial_pop.json', 'w') as f:
                json.dump(initial_pop, f, indent=4)

    return True


def load_problem_specification(save_dir, normalize, return_init_pop):
    save_dir_spec = save_dir + '/specification/'

    # load number of trials, population size, maximal number of iterations
    n_trials = int(np.loadtxt(save_dir_spec + '/n_trials.txt'))
    pop_size = int(np.loadtxt(save_dir_spec + '/pop_size.txt'))
    max_iterations = int(np.loadtxt(save_dir_spec + '/max_iterations.txt'))

    seeds = list()
    problem_dicts = list()
    init_pops = list()

    for trial in range(n_trials):
        save_dir_trial = save_dir_spec + 'trial' + str(trial) + '/'

        # load seeds
        seeds.append(float(np.loadtxt(save_dir_trial+'seed.txt')))

        # load problem dict
        with open(save_dir_trial+'problem.json', 'r') as f:
            problem_dicts.append(json.load(f))
        problem_dicts[trial]['normalize'] = normalize

        # load initial population
        if return_init_pop:
            with open(save_dir_trial+'initial_pop.json', 'r') as f:
                init_pops.append(json.load(f))
            problem_dicts[trial]['init_pop'] = init_pops[trial]

    if return_init_pop:
        return n_trials, pop_size, max_iterations, problem_dicts, seeds, init_pops

    return n_trials, pop_size, max_iterations, problem_dicts, seeds