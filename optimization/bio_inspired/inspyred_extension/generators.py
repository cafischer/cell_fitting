import numpy as np
import collections

__author__ = 'caro'


def bybounds_generator(random, n_vars, lower_bound, upper_bound):

    if np.size(lower_bound) == 1 and np.size(upper_bound) == 1:
        if isinstance(lower_bound, collections.Iterable):
            lower_bound = lower_bound[0]
        if isinstance(upper_bound, collections.Iterable):
            upper_bound = upper_bound[0]
        return [random.uniform(lower_bound, upper_bound) for i in range(n_vars)]
    elif len(lower_bound) == n_vars and len(upper_bound) == n_vars:
        return [random.uniform(lower_bound[i], upper_bound[i]) for i in range(n_vars)]
    else:
        raise ValueError('Size of upper or lower boundary is unequal to 1 or the number of variables!')