from __future__ import division
from scipy.signal import argrelmin, argrelmax
import numpy as np

__author__ = 'caro'


def get_AP_onsets(v, threshold=-45):
    """
    Returns the indices of the times where the membrane potential crossed threshold.
    :param threshold: AP threshold.
    :type threshold: float
    :return: Indices of the times where the membrane potential crossed threshold.
    :rtype: array_like
    """
    return np.nonzero(np.diff(np.sign(v-threshold)) == 2)[0]

def get_AP_max(v, AP_onset, AP_end, order=5, interval=None):
    """
    Returns the index of the local maximum of the AP between AP onset and end during dur.
    :param AP_onset: Index where the membrane potential crosses the AP threshold.
    :type AP_onset: int
    :param AP_end: Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace).
    :type AP_end: int
    :param order: Number of points to consider for determining the local maxima.
    :type order: int
    :param interval: Length of the interval during which the maximum of the AP shall occur starting from AP onset.
    :type interval: int
    :return: Index of the Maximum of the AP (None if it does not exist).
    :rtype: int
    """
    maxima = argrelmax(v[AP_onset:AP_end], order=order)[0]
    if interval is not None:
        maxima = maxima[maxima < interval]

    if np.size(maxima) == 0:
        return None
    else:
        return maxima[np.argmax(v[AP_onset:AP_end][maxima])] + AP_onset

def get_fAHP_min(v, AP_max, AP_end, order=5, interval=None):
    """
    Returns the index of the local minimum found after AP maximum.
    :param AP_max: Index of the maximum of the AP.
    :type AP_max: int
    :param AP_end: Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace).
    :type AP_end: int
    :param order: Number of points to consider for determining the local minima.
    :type order: int
    :param interval: Length of the interval during which the minimum of the fAHP shall occur starting from AP max.
    :type interval: int
    :return: Index of the Minimum of the fAHP (None if it does not exist).
    :rtype: int
    """
    minima = argrelmin(v[AP_max:AP_end], order=order)[0]
    if interval is not None:
        minima = minima[minima < interval]

    if np.size(minima) == 0:
        return None
    else:
        return minima[np.argmin(v[AP_max:AP_end][minima])] + AP_max

def get_DAP_max(v, fAHP_min, AP_end, order=5, interval=None):
    """
    Returns the index of the local maximum found after fAHP.
    :param fAHP_min: Index of the minimum of the fAHP.
    :type fAHP_min: int
    :param AP_end: Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace).
    :type AP_end: int
    :param order: Number of points to consider for determining the local minima.
    :type order: int
    :param interval: Length of the interval during which the minimum of the fAHP shall occur starting from AP max.
    :type interval: int
    :return: Index of maximum of the DAP (None if it does not exist).
    :rtype: int
    """
    maxima = argrelmax(v[fAHP_min:AP_end], order=order)[0]
    if interval is not None:
        maxima = maxima[maxima < interval]

    if np.size(maxima) == 0:
        return None
    else:
        return maxima[np.argmax(v[fAHP_min:AP_end][maxima])] + fAHP_min

def get_AP_amp(v, AP_max, vrest):
    """
    Computes the amplitude of the AP in relation to the resting potential.
    :param AP_max: Index of the maximum of the AP.
    :type AP_max: int
    :param vrest: Resting potential.
    :type vrest: float
    :return: Amplitude of the AP.
    :rtype: float
    """
    return v[AP_max] - vrest

def get_AP_width(v, t, AP_onset, AP_max, AP_end, vrest):
    """
    Computes the width at half maximum of the AP.
    :param AP_onset: Index where the membrane potential crosses the AP threshold.
    :type AP_onset: int
    :param AP_max: Index of the maximum of the AP.
    :type AP_max: int
    :param AP_end: Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace).
    :type AP_end: int
    :param vrest: Resting potential.
    :type vrest: float
    :return: AP width at half maximum.
    :rtype: float
    """
    halfmax = (v[AP_max] - vrest)/2
    width1 = np.argmin(np.abs(v[AP_onset:AP_max]-vrest-halfmax)) + AP_onset
    width2 = np.argmin(np.abs(v[AP_max:AP_end]-vrest-halfmax)) + AP_max
    return t[width2] - t[width1]

def get_DAP_amp(v, DAP_max, vrest):
    """
    Computes the amplitude of the DAP in relation to the resting potential.
    :param DAP_max: Index of maximum of the DAP.
    :type DAP_max: int
    :param vrest: Resting potential.
    :type vrest: float
    :return: Amplitude of the DAP.
    :rtype: float
    """
    return v[DAP_max] - vrest

def get_DAP_deflection(v, fAHP_min, DAP_max):
    """
    Computes the deflection of the DAP (the height of the depolarization in relation to the minimum of the fAHP).
    :param fAHP_min: Index of the Minimum of the fAHP.
    :type fAHP_min: int
    :param DAP_max: Index of maximum of the DAP.
    :type DAP_max: int
    :return: Deflection of the DAP.
    :rtype: float
    """
    return np.abs(v[DAP_max] - v[fAHP_min])

def get_DAP_width(v, t, fAHP_min, DAP_max, AP_end, vrest):
    """
    Width of the DAP (distance between the time point of the minimum of the fAHP and the time point where the
    downhill side of the DAP is closest to the half amplitude of the minimum of the fAHP).
    :param fAHP_min: Index of the Minimum of the fAHP
    :type fAHP_min: int
    :param DAP_max: Index of maximum of the DAP.
    :type DAP_max: int
    :param AP_end: Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace)
    :type AP_end: int
    :param vrest: Resting potential.
    :type vrest: float
    :return: Width of the DAP.
    :rtype: float
    """
    halfmax = (v[fAHP_min] - vrest)/2
    halfwidth = np.nonzero(np.diff(np.sign(v[DAP_max:AP_end]-vrest-halfmax)) == -2)[0][0] + DAP_max
    return t[halfwidth] - t[fAHP_min]

def get_v_rest(v, i_inj):
    """
    Computes the resting potential as the mean of the voltage starting at 0 until current is injected.

    :param i_inj: Injected current (nA).
    :type i_inj: array_like
    :return: Resting potential (mean of the voltage trace).
    :rtype: float
    """
    nonzero = np.nonzero(i_inj)[0]
    if len(nonzero) == 0:
        to_current = -1
    else:
        to_current = nonzero[0]-1
    return np.mean(v[0:to_current])


def get_inputresistance(v, i_inj):
    """Computes the input resistance. Assumes step current protocol: 0 current for some time, step to x current long
    enough to obtain the steady-state voltage.

    :param v: Voltage (mV) from the step current experiment.
    :type v: array_like
    :param i_inj: Injected current (nA).
    :type i_inj: array_like
    :return: Input resistance (MOhm).
    :rtype: float
    """
    step = np.nonzero(i_inj)[0]
    idx_step_start = step[0]
    idx_step_half = int(idx_step_start + np.round(len(step)/2.0))
    idx_step_end = step[-1]

    vrest = get_v_rest(v, i_inj)

    vstep = np.mean(v[idx_step_half:idx_step_end])  # start at the middle of the step to get the steady-state voltage

    return (vstep - vrest) / i_inj[idx_step_start]


def get_AP_start_end(v, threshold=-45, n=0):
    AP_onsets = get_AP_onsets(v, threshold)
    if len(AP_onsets) < n+1:
        return None, None
    else:
        AP_onset = AP_onsets[n]
        if len(AP_onsets) < n+2:
            AP_end = -1
        else:
            AP_end = AP_onsets[n+1]
        return AP_onset, AP_end


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as pl

    # # test on experimental data
    data_dir = '../data/2015_08_26b/raw/rampIV/3.0(nA).csv'
    data = pd.read_csv(data_dir)
    v_exp = np.array(data.v)
    i_exp = np.array(data.i)
    t_exp = np.array(data.t)
    dt_exp = t_exp[1]-t_exp[0]

    vrest = get_v_rest(v_exp, i_exp)
    AP_onsets = get_AP_onsets(v_exp, threshold=-30)
    AP_onset = AP_onsets[0]
    AP_end = -1

    AP_max = get_AP_max(v_exp, AP_onset, AP_end, interval=1/dt_exp)
    fAHP_min = get_fAHP_min(v_exp, AP_max, AP_end, interval=5/dt_exp)
    DAP_max = get_DAP_max(v_exp, fAHP_min, AP_end, interval=10/dt_exp)

    AP_amp = get_AP_amp(v_exp, AP_max, vrest)
    AP_width = get_AP_width(v_exp, t_exp, AP_onset, AP_max, AP_end, vrest)
    DAP_amp = get_DAP_amp(v_exp, DAP_max, vrest)
    DAP_deflection = get_DAP_deflection(v_exp, DAP_max, fAHP_min)
    DAP_width = get_DAP_width(v_exp, t_exp, fAHP_min, DAP_max, AP_end, vrest)
    print 'AP amplitude: ' + str(AP_amp) + ' (mV)'
    print 'AP width: ' + str(AP_width) + ' (ms)'
    print 'DAP amplitude: ' + str(DAP_amp) + ' (mV)'
    print 'DAP deflection: ' + str(DAP_deflection) + ' (mV)'
    print 'DAP width: ' + str(DAP_width) + ' (ms)'

    pl.figure()
    pl.plot(t_exp, v_exp, 'k', label='V')
    pl.plot(t_exp[AP_onsets], v_exp[AP_onsets], 'or', label='AP onsets')
    pl.plot(t_exp[AP_max], v_exp[AP_max], 'ob', label='AP maximum')
    pl.plot(t_exp[fAHP_min], v_exp[fAHP_min], 'oy', label='fAHP minimum')
    pl.plot(t_exp[DAP_max], v_exp[DAP_max], 'og', label='DAP maximum')
    pl.legend()
    pl.show()

    data_dir = '../data/2015_08_26b/raw/rampIV/3.0(nA).csv'
    data = pd.read_csv(data_dir)
    v_step = np.array(data.v)
    i_step = np.array(data.i)
    t_step = np.array(data.t)
    dt_step = t_step[1]-t_step[0]

    rin = get_inputresistance(v_step, i_step)
    print 'Input resistance: ' + str(rin) + ' (MOhm)'