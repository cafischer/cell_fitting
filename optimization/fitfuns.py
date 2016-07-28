import numpy as np
from statistics.analyze_APs import *

__author__ = 'caro'


def get_v(v, t, i_inj):
    return [v]


def get_APamp_fAHPmin_DAPamp(v, t, i_inj):
    dt = t[1] - t[0]
    vrest = get_vrest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v)
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


def get_vrest_APamp_fAHPmin_DAPamp(v, t, i_inj):
    dt = t[1] - t[0]
    vrest = get_vrest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v)
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

def get_vrest_APampwidthtime(v, t, i_inj):
    dt = t[1] - t[0]
    vrest = get_vrest(v, i_inj)
    AP_onset, AP_end = get_AP_start_end(v)
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