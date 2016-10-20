import numpy as np
from statistics.analyze_APs import *

__author__ = 'caro'


def get_v(v, t, i_inj, args):
    return [v]


def get_APamp(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    vrest = get_vrest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return [None]
    AP_max = get_AP_max(v, AP_onset, AP_end, interval=3/dt)
    if AP_max is None:
        return [None]
    AP_amp = get_AP_amp(v, AP_max, vrest)
    if AP_amp is None:
        return [None]
    return [AP_amp]


def get_APwidth(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    vrest = get_vrest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return [None]
    AP_max = get_AP_max(v, AP_onset, AP_end, interval=3/dt)
    if AP_max is None:
        return [None]
    AP_width = get_AP_width(v, t, AP_onset, AP_max, AP_end, vrest)
    if AP_width is None:
        return [None]
    return [AP_width]


def get_APtime(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return [None]
    AP_max = get_AP_max(v, AP_onset, AP_end, interval=3/dt)
    if AP_max is None:
        return [None]
    return [AP_max * dt]


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

    AP_time = get_APtime(v, t, i_inj, args)[0]
    if AP_time is None:
        return [None]

    inside_shift = check_inside_shift(AP_time / dt, APtime_data, shift)
    inside_window = check_inside_window(AP_time / dt, len(v), window_before, window_after)

    if inside_shift and inside_window:  # use APtime as reference
        window_start = int(np.round(AP_time / dt - window_before, 0))
        window_end = int(np.round(AP_time / dt + window_after, 0))
        return [v[window_start: window_end]]
    else:
        return [None]


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
        return [v[window_start: window_end]]
    else:
        return [None]


def shifted_best(v, t, i_inj, args):
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
        window_start = int(np.round(AP_time / dt - window_before, 0))
        window_end = int(np.round(AP_time / dt + window_after, 0))
    else:
        AP_onsets = get_AP_onsets(v, threshold)  # if existing use second spike as reference
        if len(AP_onsets) > 1:
            inside_window = check_inside_window(AP_onsets[1], len(v), window_before, window_after)
            inside_shift = check_inside_shift(AP_onsets[1], APtime_data, shift)
            if inside_window and inside_shift:
                window_start = int(np.round(AP_onsets[1] - window_before, 0))
                window_end = int(np.round(AP_onsets[1] + window_after, 0))
            else:
                window_start = int(np.round(APtime_data - window_before, 0))  # else use APtime from data as reference
                window_end = int(np.round(APtime_data + window_after, 0))
        else:
            window_start = int(np.round(APtime_data - window_before, 0))  # else use APtime from data as reference
            window_end = int(np.round(APtime_data + window_after, 0))

    return [v[window_start: window_end]]


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


def get_APamp_fAHPmin_DAPamp(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    vrest = get_vrest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return None, None, None
    AP_max = get_AP_max(v, AP_onset, AP_end, interval=3/dt)
    if AP_max is None:
        return None, None, None
    AP_amp = get_AP_amp(v, AP_max, vrest)
    fAHP_min = get_fAHP_min(v, AP_max, AP_end, order=5, interval=3/dt)
    fAHP_amp = v[fAHP_min]-vrest
    if fAHP_min is None:
        return None, None, None
    DAP_max = get_DAP_max(v, fAHP_min, AP_end, order=5, interval=5/dt)
    if DAP_max is None:
        return None, None, None
    DAP_amp = get_DAP_amp(v, DAP_max, vrest)

    return AP_amp, fAHP_amp, DAP_amp


def get_vrest_APamp_fAHPmin_DAPamp(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    vrest = get_vrest(v, i_inj)
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

def get_vrest_APampwidthtime(v, t, i_inj, args):
    threshold = args.get('threshold', -45)
    dt = t[1] - t[0]
    vrest = get_vrest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v, threshold)
    if AP_onset is None or AP_end is None:
        return None, None, None, None
    AP_max = get_AP_max(v, AP_onset, AP_end, interval=3/dt)
    if AP_max is None:
        return None, None, None, None
    time_AP_max = AP_max * dt
    AP_amp = get_AP_amp(v, AP_max, vrest)
    AP_width = get_AP_width(v, t, AP_onset, AP_max, AP_end, vrest)

    return vrest, AP_amp, AP_width, time_AP_max


def get_parts_restAPDAPrest(v, t, i_inj):
    dt = t[1] - t[0]
    i_inj_start = np.nonzero(i_inj)[0][0]
    AP_end = int(np.round(13/dt, 0))
    DAP_end = int(np.round(35/dt, 0))
    return v[:i_inj_start], v[i_inj_start+1:AP_end], v[AP_end:DAP_end], v[DAP_end:]


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