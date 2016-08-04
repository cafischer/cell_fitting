import pandas as pd
import numpy as np
import functools
from scipy.optimize import minimize

from optimization.scipy_based import numerical_gradient

__author__ = 'caro'


def optimize(initial_candidate, method, method_type, method_args, problem, max_iterations):

    # list for storage of candidates
    candidates = [initial_candidate]

    def store_candidates(candidate):
        candidates.append(candidate)

    # function to optimize and derivative
    fun = functools.partial(problem.evaluate, args=None)
    if method_type == 'gradient_based':
        # use numerical derivative
        funprime = functools.partial(numerical_gradient, f=fun, method='central')
    else:
        funprime = None

    # optimize
    if method_args is None:
        method_args = {}

    minimize(fun, initial_candidate, method=method, jac=funprime, callback=store_candidates,
             options={'maxiter': max_iterations-1, 'disp': False}, **method_args)
    return candidates


def save_to_individuals_file(candidates, max_iterations, pop_size, problem, individuals_file):

    # generate data frame
    generation = np.repeat(range(max_iterations+1), pop_size)
    number = range(pop_size) * (max_iterations+1)
    candidate = [str(np.array(c)) for c in candidates]
    fitness = [problem.evaluate(c, None) for c in candidates]

    individuals_data = pd.DataFrame(data={'generation': generation, 'number': number, 'fitness': np.array(fitness),
                                          'candidate': np.array(candidate)})

    # sort according to fitness inside generations
    individuals_data = individuals_data.groupby('generation').apply(lambda x: x.sort_values(['fitness']))
    individuals_data.number = range(pop_size) * (max_iterations+1)  # set again because sorting screwed it up

    # save individuals data
    individuals_data.to_csv(individuals_file, header=True, index=False)


def optimize_scipy_based(pop_size, max_iterations, method, method_type, method_args, problem, individuals_file):

    candidates = np.zeros((max_iterations+1, pop_size), dtype=object)  # +1 because of initial population

    if 'bounds' in method_args:
        method_args = {'bounds': zip(problem.lower_bound, problem.upper_bound)}

    for i in range(pop_size):
        # run optimization
        candidates_tmp = optimize(problem.init_pop.next(), method, method_type, method_args, problem,
                                  max_iterations)

        # if stopped before max_iterations, fill up by last candidate
        candidates[:, i] = candidates_tmp + [candidates_tmp[-1]] * ((max_iterations+1) - len(candidates_tmp))

    # flatten candidates for later saving
    candidates = candidates.flatten()

    # save candidates into individuals file
    save_to_individuals_file(candidates, max_iterations, pop_size, problem, individuals_file)


