import numpy as np


def init_nan(shape):
    x = np.empty(shape)
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


def convert_from_unit(prefix, x):
    """
    Converts x from unit prefix to base unit.
    :param prefix: Prefix (implemented are 'T', 'M', da', 'd', 'c', 'm', 'u', 'n', 'p').
    :type prefix:str
    :param x: Quantity to convert.
    :type x: array_like
    :return: Converted quantity.
    :rtype: array_like
    """
    if prefix == 'T':
        return x * 1e12
    elif prefix == 'M':
        return x * 1e6
    elif prefix == 'h':
        return x * 1e2
    elif prefix == 'da':
        return x * 1e1
    elif prefix == 'd':
        return x * 1e-1
    elif prefix == 'c':
        return x * 1e-2
    elif prefix == 'm':
        return x * 1e-3
    elif prefix == 'u':
        return x * 1e-6
    elif prefix == 'n':
        return x * 1e-9
    elif prefix == 'p':
        return x * 1e-12
    else:
        raise ValueError('No valid prefix!')


def convert_to_unit(prefix, x):
    """
    Converts x from base unit to unit prefix.
    :param prefix: Prefix (implemented are 'T', 'M', da', 'd', 'c', 'm', 'u', 'n', 'p').
    :type prefix:str
    :param x: Quantity to convert.
    :type x: array_like
    :return: Converted quantity.
    :rtype: array_like
    """
    if prefix == 'T':
        return x * 1e-12
    elif prefix == 'M':
        return x * 1e-6
    elif prefix == 'h':
        return x * 1e-2
    elif prefix == 'da':
        return x * 1e-1
    elif prefix == 'd':
        return x * 1e1
    elif prefix == 'c':
        return x * 1e2
    elif prefix == 'm':
        return x * 1e3
    elif prefix == 'u':
        return x * 1e6
    elif prefix == 'n':
        return x * 1e9
    elif prefix == 'p':
        return x * 1e12
    else:
        raise ValueError('No valid prefix!')