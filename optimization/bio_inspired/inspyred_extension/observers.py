from time import time

__author__ = 'caro'


def individuals_observer(population, num_generations, num_evaluations, args):

    try:
        individuals_file = args['individuals_file']
    except KeyError:
        individuals_file = open('inspyred-individuals-file-{0}.csv'.format(time.strftime('%m%d%Y-%H%M%S')), 'w')

    if num_generations == 0:
        individuals_file.write('{0}, {1}, {2}, {3}\n'.format('generation', 'id', 'fitness', 'candidate'))

    population = sorted(population, key=lambda p: p.fitness)
    for i, p in enumerate(population):
        individuals_file.write('{0}, {1}, {2}, {3}\n'.format(num_generations, i, p.fitness,
                                                str(p.candidate).replace(',', '').replace('[', '').replace(']', '')))
    individuals_file.flush()