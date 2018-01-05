import numpy as np


def init_nan(shape):
    x = np.zeros(shape)
    x[:] = np.nan
    return x


def merge_dicts(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict


def norm(x, lower_bound, upper_bound):
    return [(x[i] - lower_bound[i]) / (upper_bound[i] - lower_bound[i]) for i in range(len(x))]


def unnorm(x, lower_bound, upper_bound):
    return [x[i] * (upper_bound[i] - lower_bound[i]) + lower_bound[i] for i in range(len(x))]