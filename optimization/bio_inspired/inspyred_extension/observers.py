import numpy as np
import pandas as pd

__author__ = 'caro'


def individuals_observer(population, num_generations, num_evaluations, args):
    """
    Stores the current generation in args['individuals_file'] with columns ['generation', 'number', 'fitness',
    'candidate'].
    Generation: Generation of the candidate
    Number: Index of the candidate in this generation
    Fitness: Fitness of the candidate
    Candidate: String representation of the current candidate
    :param population: Actual population.
    :type population: array_like
    :param num_generations: Actual generation.
    :type num_generations: int
    :param num_evaluations: Actual number of evaluations.
    :type num_evaluations: int
    :param args: Additional arguments. Should contain a file object under the keyword 'individuals_file'
    :type args: dict
    """

    generation = np.zeros(len(population))
    number = np.zeros(len(population))
    fitness = np.zeros(len(population))
    candidates = np.zeros(len(population), dtype=object)

    for i, p in enumerate(population):
        generation[i] = num_generations
        number[i] = i
        fitness[i] = p.fitness
        candidates[i] = str(p.candidate)

    # for sorting according to fitness
    idx = np.argsort(fitness)

    # create DataFrame for this generation
    individuals_file = pd.DataFrame(data={'generation': generation, 'number': number, 'fitness': fitness[idx],
                                          'candidate': candidates[idx]})

    # save DataFrame from this generation
    if num_generations == 0:
        individuals_file.to_csv(args['individuals_file'], header=True, index=False)
    else:
        individuals_file.to_csv(args['individuals_file'], header=False, index=False)