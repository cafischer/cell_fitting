import numpy as np
from scipy.spatial.distance import euclidean
#from fastdtw import fastdtw
from math import *
import sys

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


def meanabs_error(a, b):
    return np.sum(np.abs(np.array(a) - np.array(b)))


def maxabs_error(a, b):
    return np.max(np.abs(np.array(a) - np.array(b)))


#def dtw(x, y):
#    distance, path = fastdtw(x, y, dist=euclidean)
#    return distance


def dtw_with_window(A, B, window=500, d=euclidean):
    # create the cost matrix
    A, B = np.array(A), np.array(B)
    M, N = len(A), len(B)
    cost = sys.maxint * np.ones((M, N))

    # initialize the first row and column
    cost[0, 0] = d(A[0], B[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(A[i], B[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(A[0], B[j])
    # fill in the rest of the matrix
    for i in range(1, M):
        for j in range(max(1, i - window), min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(A[i], B[j])

    # find the optimal path
    n, m = N - 1, M - 1
    path = []

    while (m, n) != (0, 0):
        path.append((m, n))
        m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key=lambda x: cost[x[0], x[1]])

    path.append((0, 0))
    return cost[-1, -1]  #, path