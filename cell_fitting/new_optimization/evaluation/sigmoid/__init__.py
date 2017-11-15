import numpy as np


def sig(l_a, u_a, a, x):
    return l_a + (u_a - l_a) / (1 + np.exp(-a * x))


def sig_upper_half(l_a, u_a, a, x):
    sigmoid = sig(l_a - np.abs(l_a-u_a), u_a, a, x)
    return np.concatenate((np.ones(int(np.floor(len(x)/2)))*l_a, sigmoid[int(np.ceil(len(x)/2)):]))