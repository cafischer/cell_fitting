from time import time
from optimization.bio_inspired.inspyred_extension import *

__author__ = 'caro'


def individuals_observer(population, num_generations, num_evaluations, args):

    try:
        individuals_file = args['individuals_file']
    except KeyError:
        individuals_file = open('inspyred-individuals-file-{0}.csv'.format(time.strftime('%m%d%Y-%H%M%S')), 'w')

    if num_generations == 0:
        individuals_file.write('{0},{1},{2},{3}\n'.format('generation', 'id', 'fitness', 'candidate'))

    population = sorted(enumerate(population), key=lambda x: x[1].fitness)
    for id, p in population:
        individuals_file.write('{0},{1},{2},{3}\n'.format(num_generations, id, p.fitness,
                                                str(p.candidate).replace(',', '').replace('[', '').replace(']', '')))
    individuals_file.flush()


def collect_observer(population, num_generations, num_evaluations, args):

    population = sorted(population, key=lambda p: p.fitness)
    for p in population:
        args['individuals'].append([num_generations, args['id'], p.fitness,
                            str(p.candidate).replace(',', '').replace('[', '').replace(']', '')])


def normalize_observer(observer, lower_bounds, upper_bounds):

    def normalized_observer(population, num_generations, num_evaluations, args):
        population_unnormed = unnorm_population(population,lower_bounds, upper_bounds)
        return observer(population_unnormed, num_generations, num_evaluations, args)

    return normalized_observer