from util import *


__author__ = 'caro'


def get_random_numbers_in_bounds(random, lower_bounds, upper_bounds, args):
    assert(len(lower_bounds) == len(upper_bounds))
    assert(all([lower_bounds[i] <= upper_bounds[i] for i in range(len(lower_bounds))]))
    return [random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]


def normalize_generator(generator, lower_bounds, upper_bounds):

    def generator_normalized(random, args):
        return norm(generator(random, args=args), lower_bounds, upper_bounds)  # args by keyword in case
                                                                               # functools.partial is used
    return generator_normalized