import numpy as np
import matplotlib.pyplot as pl
from nrn_wrapper import vclamp

__author__ = 'caro'


def extract_simulation_params(data, sec=('soma', None), celsius=35, pos_i=0.5, pos_v=0.5):
    """
    Uses the experimental data and additional arguments to extract the simulation parameters.

    :param data: Dataframe containing the columns: v (membrane potential), t (Time), i (injected current).
    :type data: pandas.DataFrame
    :param sec: Section to stimulate and record from. First argument name, second argument index (None for no index).
    :type sec: tuple
    :param celsius: Temperatur during simulation (affects ion channel kinetics).
    :type celsius: float
    :param pos_i: Position of the stimulating electrode (value between 0 and 1).
    :type pos_i: float
    :param pos_v: Position of the recording electrode (value between 0 and 1).
    :type pos_v: float
    :return: Simulation parameter
    :rtype: dict
    """
    # load experimental data and simulation parameters
    tstop = data.t.values[-1]
    dt = data.t.values[1] - data.t.values[0]
    v_init = data.v.values[0]
    i_inj = data.i.values
    pos_i = pos_i
    pos_v = pos_v
    sec = sec
    return {'i_inj': i_inj, 'v_init': v_init, 'tstop': tstop, 'dt': dt, 'pos_i': pos_i,
                                  'pos_v': pos_v, 'sec': sec, 'celsius': celsius}


def currents_given_v(v, t, sec, channel_list, ion_list, celsius, plot=False):
    """
    Records currents from sec elicited by clamping v.
    :param v: Voltage to clamp.
    :type v: array_like
    :param t: Time corresponding to v.
    :type t: array_like
    :param sec: Section where to apply the voltage clamp.
    :type sec: Section
    :param channel_list: List of ion channels that shall be measured.
    :type channel_list: array_like
    :param ion_list: List of ions that flow to the respective ion channel in the list.
    :type ion_list: array_like
    :param plot: If true, each current trace is plotted.
    :type plot: bool
    :return: Measured current traces.
    :rtype: array_like
    """

    # record currents
    currents = np.zeros(len(channel_list), dtype=object)
    for i in range(len(channel_list)):
        currents[i] = sec.record_from(channel_list[i], 'i'+ion_list[i], pos=.5)

    # apply vclamp
    vclamp(v, t, sec, celsius)

    # convert current traces to array
    for i in range(len(channel_list)):
        currents[i] = np.array(currents[i])

        # plot current traces
        if plot:
            pl.plot(t, currents[i], 'k')
            pl.ylabel('Current (mA/cm2)')
            pl.xlabel('Time (ms)')
            pl.title(channel_list[i])
            pl.show()

    return currents