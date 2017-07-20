import pandas as pd
import numpy as np

__author__ = 'caro'


def correct_baseline(y, vrest=None, v_rest_change=None):
    if vrest is not None:
        y = y - (y[0] - vrest)
    if v_rest_change is not None:
        y += v_rest_change
    return y


def power_of_2(n):
    power_of_2 = 1
    while power_of_2 <= n:
        if power_of_2 == n:
            return True
        power_of_2 *= 2
    return False


def change_dt(dt_new, data):
    """
    Just does linear interpolation on data.
    Note: Only save to use new time steps: dt_new = dt_old / 2**x. So that important points in the traces are preserved.
    :param dt_new:
    :type dt_new:
    :param data:
    :type data:
    :return:
    :rtype:
    """
    dt_old = data.t.values[1] - data.t.values[0]
    if dt_new < dt_old: assert power_of_2(dt_old/dt_new)

    t = np.arange(0, data.t.values[-1]+dt_new, dt_new)
    i = np.interp(t, data.t.values, data.i.values)
    v = np.interp(t, data.t.values, data.v.values)

    return pd.DataFrame({'t': t, 'i': i, 'v': v})