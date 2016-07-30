import numpy as np

__author__ = 'caro'


def bybounds_generator(random, n_vars, lower_bound, upper_bound):

    if np.size(lower_bound) == n_vars and np.size(upper_bound) == n_vars:
        return np.array([random.uniform(lower_bound[i], upper_bound[i]) for i in range(n_vars)])
    elif np.size(lower_bound) == 1 and np.size(upper_bound) == 1:
        return np.array([random.uniform(lower_bound, upper_bound) for i in range(n_vars)])
    else:
        raise ValueError('Size of upper or lower boundary is unequal to 1 or the number of variables!')