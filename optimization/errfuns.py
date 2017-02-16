import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

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
    return np.sqrt(np.sum((np.array(a) - np.array(b))**2) / np.size(a))

def maxabs(a, b):
    return np.max(np.abs(np.array(a) - np.array(b)))

def dtw(x, y):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance