import pandas as pd
import numpy as np

__author__ = 'caro'


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
    assert power_of_2(dt_old/dt_new)

    t = np.arange(0, data.t.values[-1]+dt_new, dt_new)
    i = np.interp(t, data.t.values, data.i.values)
    v = np.interp(t, data.t.values, data.v.values)
    sec = np.zeros(len(t), dtype=object)
    sec[0] = data.sec.values[0]
    sec[1] = data.sec.values[1]
    sec[1:] = np.nan

    return pd.DataFrame({'t': t, 'i': i, 'v': v, 'sec': sec})