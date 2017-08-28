import inspyred
from cell_fitting.util import *

__author__ = 'caro'


def unnorm_population(population, lower_bound, upper_bound):
    population_unnormed = list()
    for i, p in enumerate(population):
        individual = inspyred.ec.Individual()  # explicitly copy because candidate setter changes fitness to None
        individual.candidate = unnorm(p.candidate, lower_bound, upper_bound)
        individual.fitness = p.fitness
        individual.birthdate = p.birthdate
        individual.maximize = p.maximize
        population_unnormed.append(individual)
    return population_unnormed