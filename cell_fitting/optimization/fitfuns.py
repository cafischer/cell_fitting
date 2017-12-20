import numpy as np
from cell_characteristics.analyze_APs import *
import scipy

__author__ = 'caro'


def get_v(v, t, i_inj, args):
    return v


def get_DAP(v, t, i_inj, args):
    dt = t[1] - t[0]
    DAP_start = int(round(13.5 / dt))
    DAP_end = int(round(160.0 / dt))
    v_DAP = v[DAP_start:DAP_end]
    return v_DAP


def get_n_spikes(v, t, i_inj, args):
    threshold = args.get('threshold', -10)
    AP_onsets = get_AP_onset_idxs(v, threshold)
    return len(AP_onsets)


def phase_hist(v, t, i_inj, args):
    v_min = args['v_min']
    v_max = args['v_max']
    dvdt_min = args['dvdt_min']
    dvdt_max = args['dvdt_max']
    bins_v = args['bins_v']
    bins_dvdt = args['bins_dvdt']

    dt = t[1] - t[0]
    dvdt = np.concatenate((np.array([(v[1]-v[0])/dt]), np.diff(v) / dt))

    H, v_range, dvdt_range = np.histogram2d(v, dvdt, bins=[bins_v, bins_dvdt],
                                            range=[[v_min, v_max], [dvdt_min, dvdt_max]])
    return H


def get_APamp(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    vrest = get_v_rest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return None
    AP_max = get_AP_max_idx(v, AP_onset, AP_end, interval=1/dt)
    if AP_max is None:
        return None
    AP_amp = get_AP_amp(v, AP_max, vrest)
    if AP_amp is None:
        return None
    return AP_amp


def get_APwidth(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    vrest = get_v_rest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return None
    AP_max = get_AP_max_idx(v, AP_onset, AP_end, interval=1/dt)
    if AP_max is None:
        return None
    AP_width = get_AP_width(v, t, AP_onset, AP_max, AP_end, vrest)
    if AP_width is None:
        return None
    return AP_width


def get_APtime(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return None
    AP_max = get_AP_max_idx(v, AP_onset, AP_end, interval=1/dt)
    if AP_max is None:
        return None
    return AP_max * dt


def penalize_not1AP(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    penalty = args.get('penalty', 100)
    dt = t[1] - t[0]
    AP_onsets = get_AP_onset_idxs(v, threshold)
    if len(AP_onsets) == 1:
        return 0
    return penalty


def shifted_AP(v, t, i_inj, args):
    dt = t[1] - t[0]
    APtime_data = args['APtime'] / dt
    shift = args['shift'] / dt
    window_before = args['window_before'] / dt
    window_after = args['window_after'] / dt

    if APtime_data - window_before < 0 or APtime_data + window_after >= len(v):
        raise ValueError('AP data is not inside window')

    AP_time = get_APtime(v, t, i_inj, args)
    if AP_time is None:
        return None

    inside_shift = check_inside_shift(AP_time / dt, APtime_data, shift)
    inside_window = check_inside_window(AP_time / dt, len(v), window_before, window_after)

    if inside_shift and inside_window:  # use APtime as reference
        window_start = int(np.round(AP_time / dt - window_before, 0))
        window_end = int(np.round(AP_time / dt + window_after, 0))
        return v[window_start: window_end]
    else:
        return None


def shifted_max(v, t, i_inj, args):
    dt = t[1] - t[0]
    APtime_data = args['APtime'] / dt
    shift = args['shift'] / dt
    window_before = args['window_before'] / dt
    window_after = args['window_after'] / dt

    if APtime_data - window_before < 0 or APtime_data + window_after >= len(v):
        raise ValueError('AP data is not inside window')

    AP_time = get_APtime(v, t, i_inj, args)[0]
    if AP_time is None:  # try with the maximum inside the shift region
        idx0 = np.max([0, APtime_data - shift])
        idx1 = np.min([len(v), APtime_data + shift])
        AP_time = t[idx0 + np.argmax(v[idx0: idx1])]

    inside_shift = check_inside_shift(AP_time / dt, APtime_data, shift)
    inside_window = check_inside_window(AP_time / dt, len(v), window_before, window_after)

    if inside_shift and inside_window:  # use APtime as reference
        window_start = int(np.round(AP_time / dt - window_before, 0))
        window_end = int(np.round(AP_time / dt + window_after, 0))
        return v[window_start: window_end]
    else:
        return None


def check_inside_shift(AP_time, APtime_data, shift):
    inside_shift = True
    if np.abs(APtime_data - AP_time) > shift:
        inside_shift = False
    return inside_shift


def check_inside_window(AP_time, max_len, window_before, window_after):
    inside_window = True
    window_start = int(np.round(AP_time - window_before, 0))
    window_end = int(np.round(AP_time + window_after, 0))
    if window_start < 0 or window_end >= max_len:
        inside_window = False
    return inside_window


def get_vrest(v, t, i_inj, args):
    return get_v_rest(v, i_inj)


def get_fAHPamp(v, t, i_inj, args):
    threshold = args.get('threshold', -20)
    dt = t[1] - t[0]
    vrest = get_v_rest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return None
    AP_max = get_AP_max_idx(v, AP_onset, AP_end, interval=1/dt)
    if AP_max is None:
        return None
    fAHP_min = get_fAHP_min_idx(v, AP_max, AP_end, interval=5/dt)
    fAHP_amp = v[fAHP_min]-vrest
    return fAHP_amp


def get_DAPamp(v, t, i_inj, args):
    threshold = args.get('threshold', -20)
    dt = t[1] - t[0]
    vrest = get_v_rest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return None
    AP_max = get_AP_max_idx(v, AP_onset, AP_end, interval=1/dt)
    if AP_max is None:
        return None
    fAHP_min = get_fAHP_min_idx(v, AP_max, AP_end, interval=5/dt)
    if fAHP_min is None:
        return None
    DAP_max = get_DAP_max_idx(v, fAHP_min, AP_end, interval=10/dt)
    if DAP_max is None:
        return None
    DAP_amp = get_DAP_amp(v, DAP_max, vrest)
    return DAP_amp


def get_DAPdeflection(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return None
    AP_max = get_AP_max_idx(v, AP_onset, AP_end, interval=1/dt)
    if AP_max is None:
        return None
    fAHP_min = get_fAHP_min_idx(v, AP_max, AP_end, interval=5/dt)
    if fAHP_min is None:
        return None
    DAP_max = get_DAP_max_idx(v, fAHP_min, AP_end, interval=10/dt)
    if DAP_max is None:
        return None
    DAP_deflection = get_DAP_deflection(v, fAHP_min, DAP_max)
    return DAP_deflection


def get_DAPwidth(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    vrest = get_v_rest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return None
    AP_max = get_AP_max_idx(v, AP_onset, AP_end, interval=1/dt)
    if AP_max is None:
        return None
    fAHP_min = get_fAHP_min_idx(v, AP_max, AP_end, interval=5/dt)
    if fAHP_min is None:
        return None
    DAP_max = get_DAP_max_idx(v, fAHP_min, AP_end, interval=10/dt)
    if DAP_max is None:
        return None
    DAP_width = get_DAP_width(v, t, fAHP_min, DAP_max, AP_end, vrest)
    return DAP_width


def get_DAPtime(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    vrest = get_v_rest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return None
    AP_max = get_AP_max_idx(v, AP_onset, AP_end, interval=1/dt)
    if AP_max is None:
        return None
    fAHP_min = get_fAHP_min_idx(v, AP_max, AP_end, interval=5/dt)
    if fAHP_min is None:
        return None
    DAP_max = get_DAP_max_idx(v, fAHP_min, AP_end, interval=10/dt)
    if DAP_max is None:
        return None
    DAP_time_from_AP = t[DAP_max] - t[AP_max]
    return DAP_time_from_AP


def impedance(v, i_inj, dt, f_range):
    """
    Computes the impedance (impedance = fft(v) / fft(i)) for a given range of frequencies.

    :param v: Membrane potential (mV)
    :type v: array
    :param i_inj: Current (nA)
    :type i_inj: array
    :param dt: Time step.
    :type dt: float
    :param f_range: Boundaries of the frequency interval.
    :type f_range: list
    :return: Impedance (MOhm)
    :rtype: array
    """

    # FFT of the membrance potential and the input current
    fft_i = np.fft.fft(i_inj)
    fft_v = np.fft.fft(v)
    freqs = np.fft.fftfreq(v.size, d=dt)

    # sort everything according to the frequencies
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fft_i = fft_i[idx]
    fft_v = fft_v[idx]

    # calculate the impedance
    imp = np.abs(fft_v/fft_i)

    # index with frequency range
    idx1 = np.argmin(np.abs(freqs-f_range[0]))
    idx2 = np.argmin(np.abs(freqs-f_range[1]))

    return imp[idx1:idx2], freqs[idx1:idx2]


def get_fAHP_min(v, t, i_inj, args):
    AP_threshold = args.get('threshold', 0)
    start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
    v_rest = np.mean(v[0:start_i_inj])
    AP_interval = 2.5  # ms (also used as interval for fAHP)
    fAHP_interval = 4.0
    AP_width_before_onset = 2  # ms
    DAP_interval = 10  # ms
    order_fAHP_min = 1.0  # ms (how many points to consider for the minimum)
    order_DAP_max = 1.0  # ms (how many points to consider for the minimum)
    min_dist_to_DAP_max = 0.5  # ms
    k_splines = 3
    s_splines = None

    fAHP_min_idx = get_spike_characteristics(v[start_i_inj:], t[start_i_inj:], ['fAHP_min_idx'], v_rest, AP_threshold,
                                         AP_interval, AP_width_before_onset, fAHP_interval, (None, None), k_splines,
                                         s_splines, order_fAHP_min, DAP_interval, order_DAP_max,
                                         min_dist_to_DAP_max, check=False)[0]
    if fAHP_min_idx is None:
        return None
    else:
        return v[start_i_inj:][fAHP_min_idx]


def get_DAP_time(v, t, i_inj, args):
    AP_threshold = args.get('threshold', 0)
    start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
    v_rest = np.mean(v[0:start_i_inj])
    AP_interval = 2.5  # ms (also used as interval for fAHP)
    fAHP_interval = 4.0
    AP_width_before_onset = 2  # ms
    DAP_interval = 10  # ms
    order_fAHP_min = 1.0  # ms (how many points to consider for the minimum)
    order_DAP_max = 1.0  # ms (how many points to consider for the minimum)
    min_dist_to_DAP_max = 0.5  # ms
    k_splines = 3
    s_splines = 0

    DAP_time = \
    get_spike_characteristics(v[start_i_inj:], t[start_i_inj:], ['DAP_time'], v_rest, AP_threshold,
                              AP_interval, AP_width_before_onset, fAHP_interval, (None, None), k_splines,
                              s_splines, order_fAHP_min, DAP_interval, order_DAP_max,
                              min_dist_to_DAP_max, check=False)[0]
    return DAP_time


def v_AP_v_DAP_DAP_time_and_diff_fAHP(data_dicts, args=None):
    is_data = args.get('is_data', False)
    fAHP_min_idx = to_idx(13, data_dicts[0]['t'][1]-data_dicts[0]['t'][0])
    if is_data:
        DAP_time = 4.94
        fAHP_diff = -1.58
    else:
        DAP_time = get_DAP_time(data_dicts[0]['v'], data_dicts[0]['t'], data_dicts[0]['i_inj'], args)

        fAHP1 = get_fAHP_min(data_dicts[1]['v'], data_dicts[1]['t'], data_dicts[1]['i_inj'], args)
        fAHP2 = get_fAHP_min(data_dicts[2]['v'], data_dicts[2]['t'], data_dicts[2]['i_inj'], args)
        if fAHP1 is None or fAHP2 is None:
            fAHP_diff = None
        else:
            fAHP_diff = (fAHP2 - fAHP1)
    return data_dicts[0]['v'][:fAHP_min_idx], data_dicts[0]['v'][fAHP_min_idx:], DAP_time, fAHP_diff


def v_AP_v_DAP_and_DAP_time(data_dicts, args=None):
    is_data = args.get('is_data', False)
    fAHP_min_idx = to_idx(13, data_dicts[0]['t'][1]-data_dicts[0]['t'][0])
    if is_data:
        DAP_time = 4.94
    else:
        DAP_time = get_DAP_time(data_dicts[0]['v'], data_dicts[0]['t'], data_dicts[0]['i_inj'], args)
    return data_dicts[0]['v'][:fAHP_min_idx], data_dicts[0]['v'][fAHP_min_idx:], DAP_time


if __name__ == '__main__':

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as pl
    save_dir = '../data/toymodels/hhCell/ramp.csv'
    data = pd.read_csv(save_dir)

    dt = data.t.values[1] - data.t.values[0]
    dvdt = np.concatenate((np.array([(data.v[1]-data.v[0])/dt]), np.diff(data.v) / dt))
    args = {'v_min': np.min(data.v), 'v_max': np.max(data.v), 'dvdt_min': np.min(dvdt), 'dvdt_max': np.max(dvdt),
            'bins_v': 100, 'bins_dvdt': 100}
    H = phase_hist(np.array(data.v), np.array(data.t), np.array(data.i), args)

    pl.figure()
    pl.imshow(np.log(H[0].T), origin='lower', interpolation='none')
    pl.show()