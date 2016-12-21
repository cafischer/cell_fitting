import numpy as np
from cell_characteristics.analyze_APs import *
import scipy
from optimization.errfuns import rms

__author__ = 'caro'


def get_v(v, t, i_inj, args):
    return v


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
    AP_max = get_AP_max(v, AP_onset, AP_end, interval=3/dt)
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
    AP_max = get_AP_max(v, AP_onset, AP_end, interval=3/dt)
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
    AP_max = get_AP_max(v, AP_onset, AP_end, interval=3/dt)
    if AP_max is None:
        return None
    return AP_max * dt


def penalize_not1AP(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    penalty = args.get('penalty', 100)
    dt = t[1] - t[0]
    AP_onsets = get_AP_onsets(v, threshold)
    if len(AP_onsets) == 1:
        return 0
    return penalty


def shifted_AP(v, t, i_inj, args):
    """

    :param v:
    :type v:
    :param t:
    :type t:
    :param i_inj:
    :type i_inj:
    :param APtime_data:
    :type APtime_data:
    :param shift: in ms
    :type shift:
    :param window_size: in ms
    :type window_size:
    :return:
    :rtype:
    """
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
    """
    :param v:
    :type v:
    :param t:
    :type t:
    :param i_inj:
    :type i_inj:
    :param APtime_data:
    :type APtime_data:
    :param shift: in ms
    :type shift:
    :param window_size: in ms
    :type window_size:
    :return:
    :rtype:
    """
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


def shift_AP_max_APdata(v, t, i_inj, args):
    """
    :param v:
    :type v:
    :param t:
    :type t:
    :param i_inj:
    :type i_inj:
    :param APtime_data:
    :type APtime_data:
    :param shift: in ms
    :type shift:
    :param window_size: in ms
    :type window_size:
    :return:
    :rtype:
    """
    dt = t[1] - t[0]
    APtime_ref = None
    APtime_data = args['APtime'] / dt
    shift = args['shift'] / dt
    window_before = args['window_before'] / dt
    window_after = args['window_after'] / dt
    threshold = args.get('threshold', -45)

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
        APtime_ref = AP_time / dt
    else:
        AP_onset, AP_end = get_AP_start_end(v, threshold, n=1)
        if AP_onset is not None:
            APtime_second = get_AP_max(v, AP_onset, AP_end, interval=3/dt)
            if APtime_second is not None:
                inside_window = check_inside_window(APtime_second, len(v), window_before, window_after)
                inside_shift = check_inside_shift(APtime_second, APtime_data, shift)
                if inside_window and inside_shift:
                    APtime_ref = APtime_second

    if APtime_ref is None:
        APtime_ref = APtime_data

    window_start = int(np.round(APtime_ref - window_before, 0))  # else use APtime from data as reference
    window_end = int(np.round(APtime_ref + window_after, 0))

    return v[window_start: window_end]


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


def interpolate_fitness(v, t, i_inj, args):
    fitness_mat = args['fitness']
    p1_range = args['p1_range']
    p2_range = args['p2_range']
    candidate = args['candidate']
    return scipy.interpolate.interp2d(p1_range, p2_range, fitness_mat.T)(candidate[0], candidate[1])


def get_vrest_APamp_fAHPmin_DAPamp(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    vrest = get_v_rest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return None
    AP_max = get_AP_max(v, AP_onset, AP_end, interval=3/dt)
    if AP_max is None:
        return None
    time_AP_max = AP_max * dt
    AP_amp = get_AP_amp(v, AP_max, vrest)
    fAHP_min = get_fAHP_min(v, AP_max, AP_end, order=5, interval=3/dt)
    fAHP_amp = v[fAHP_min]-vrest
    if fAHP_min is None:
        return None
    DAP_max = get_DAP_max(v, fAHP_min, AP_end, order=5, interval=5/dt)
    if DAP_max is None:
        return None
    DAP_amp = get_DAP_amp(v, DAP_max, vrest)

    return vrest, time_AP_max, AP_amp, fAHP_amp, DAP_amp


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


def stefans_fun(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    penalty = args.get('penalty', 10)

    # different behavior depending on number of APs
    AP_onsets = get_AP_onsets(v, threshold)
    if len(AP_onsets) == 1:
        v_rest_data = args['v_rest']
        APtime_data = args['APtime']
        v_half2AP_data = args['v_half2AP']
        dt = t[1] - t[0]
        v_rest = get_v_rest(v, i_inj)
        APtime = get_APtime(v, t, i_inj, args)
        i_inj_start = np.where(i_inj > 0)[0][0]
        time_i_inj2AP = APtime - i_inj_start
        v_half2AP = v[(i_inj_start + time_i_inj2AP / 2) * dt]
        return rms(v_rest, v_rest_data) / v_rest_data + rms(APtime, APtime_data) / APtime_data \
               + rms(v_half2AP, v_half2AP_data) / v_half2AP_data
    elif len(AP_onsets) == 0:
        v_max_data = args['v_max']
        v_max = np.max(v)
        v_max /= v_max_data
        return -v_max + penalty
    else:
        v_mean_data = args['v_mean']
        v_mean = np.mean(v)
        v_mean /= v_mean_data
        return v_mean + penalty


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