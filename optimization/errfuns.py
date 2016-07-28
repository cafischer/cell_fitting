import numpy as np

__author__ = 'caro'


def rms(a, b):
    """
    Computes the root mean squared error of the two input arrays.

    :param a: Array of floats.
    :type a: array_like
    :param b: Array of floats.
    :type b: array_like
    :return: Mean quadratic error of the two input arrays.
    :rtype: float
    """
    return np.sqrt(np.sum((a - b)**2) / np.size(a))
