import inspyred
from random import Random
import pandas as pd
import shutil
import os

from utilities import merge_dicts

__author__ = 'caro'


def optimize_bio_inspired(method, method_type, method_args, problem, individuals_file, seed):

    # create pseudo random number generator
    prng = Random()
    prng.seed(seed)

    # get optimization method
    ea = getattr(getattr(inspyred, method_type), method)(prng)
    if method == 'GA':
        ea.variator = [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.gaussian_mutation]
    ea.observer = problem.observer
    ea.terminator = inspyred.ec.terminators.generation_termination

    # additional arguments for bio inspired evolution
    args = merge_dicts(method_args, {'individuals_file': individuals_file})

    # run optimization
    ea.evolve(generator=problem.generator,
              evaluator=problem.evaluator,
              maximize=problem.maximize,
              bounder=problem.bounder,
              **args)


def optimize_SA(save_dir, pop_size, method, method_type, method_args, problem, individuals_file, seed):

    # list for storing and later merging temporary individuals data
    individuals_data_tmp = list()

    # directory for temporary storage of intermediate individuals files
    save_dir_tmp = save_dir+'tmp/'
    if not os.path.exists(save_dir_tmp):
        os.makedirs(save_dir_tmp)

    for i in range(pop_size):
        with open(save_dir_tmp+'individuals_file_'+str(i)+'.csv', 'w') as individuals_file_tmp:
            optimize_bio_inspired(method, method_type, method_args, problem, individuals_file_tmp, seed)

        # load temporary individual files
        individuals_data_tmp.append(pd.read_csv(save_dir_tmp+'individuals_file_'+str(i)+'.csv'))

        # change number to number of pop_size (each individuals file represents one individual developing over generations)
        individuals_data_tmp[i].number = i

    # merge all individual files
    individuals_data = pd.concat(individuals_data_tmp, ignore_index=True)

    # sort according to fitness inside generations
    individuals_data = individuals_data.groupby('generation').apply(lambda x: x.sort_values(['fitness']))
    individuals_data.number = range(pop_size) * (method_args['max_generations']+1)  # set again because sorting screwed it up

    # save individuals file
    individuals_data.to_csv(individuals_file, header=True, index=False)

    # delete temporary files
    shutil.rmtree(save_dir_tmp)