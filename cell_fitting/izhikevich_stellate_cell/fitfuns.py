import numpy as np


def get_AP_time(v, t, i, v_peak):
    #AP_time = 11.95  # data
    idx_spikes = np.where(v == v_peak)[0]
    if len(idx_spikes) == 1:
        return idx_spikes[0] * t[1]
    return None


def get_v(v, t, i):
    return v


def get_v_DAP(v, t, i, v_peak, data_to_fit):  # TODO
    # data_to_fit =
    # idx_start = int(13.6 / t[1])
    # idx_end = int(120 / t[1])
    # data.v.values[idx_start:idx_end]
    idx_spikes = np.where(v >= v_peak)[0]
    if len(idx_spikes) == 1:
        idx_start = idx_spikes[0] + 1
        idx_end = idx_start + len(data_to_fit)
        if idx_end < len(v):
            return v[idx_start:idx_end]
    return None