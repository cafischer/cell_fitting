import numpy as np
from neuron import h
import matplotlib.pyplot as pl
h.load_file("stdrun.hoc")  # load NEURON libraries
h.cvode_active(0)  # invariable time step in NEURON

__author__ = 'caro'


def run_simulation(cell, sec, i_amp, v_init, tstop, dt, celsius=35, pos_i=0.5, pos_v=0.5):
    """
    Runs a NEURON simulation of the cell for the given parameters.

    :param i_amp: Amplitude of the injected current for all times t.
    :type i_amp: array_like
    :param v_init: Initial membrane potential of the cell.
    :type v_init: float
    :param tstop: Duration of a whole run.
    :type tstop: float
    :param dt: Time step.
    :type dt: float
    :param celsius: Temperature during the simulation (affects ion channel kinetics).
    :type celsius: float
    :param pos_i: Position of the IClamp on the Section (number between 0 and 1).
    :type pos_i: float
    :param pos_v: Position of the recording electrode on the Section (number between 0 and 1).
    :type pos_v: float
    :return: Membrane potential of the cell and time recorded at each time step.
    :rtype: tuple of three ndarrays
    """

    section = cell.substitute_section(sec[0], sec[1])

    # time
    t = np.arange(0, tstop + dt, dt)

    # insert an IClamp with the current trace from the experiment
    stim, i_vec, t_vec = section.play_current(i_amp, t, pos_i)

    # record the membrane potential
    v = section.record('v', pos_v)
    t = h.Vector()
    t.record(h._ref_t)

    # run simulation
    h.celsius = celsius
    h.v_init = v_init
    h.tstop = tstop
    h.steps_per_ms = 1 / dt  # change steps_per_ms before dt, otherwise dt not changed properly
    h.dt = dt
    h.run()

    return np.array(v), np.array(t)


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
    i_amp = data.i.values
    pos_i = pos_i
    pos_v = pos_v
    sec = sec
    return {'i_amp': i_amp, 'v_init': v_init, 'tstop': tstop, 'dt': dt, 'pos_i': pos_i,
                                  'pos_v': pos_v, 'sec': sec, 'celsius': celsius}


def vclamp(v, t, sec, celsius):

    # create SEClamp
    v_clamp = h.Vector()
    v_clamp.from_python(v)
    t_clamp = h.Vector()
    t_clamp.from_python(np.concatenate((np.array([0]), t)))  # shifted because membrane potential lags behind vclamp
    clamp = h.SEClamp(0.5, sec=sec)
    clamp.rs = 1e-15  # series resistance should be as small as possible
    clamp.dur1 = 1e9
    v_clamp.play(clamp._ref_amp1, t_clamp)

    # simulate
    h.celsius = celsius
    dt = t[1] - t[0]
    h.tstop = t[-1]
    h.steps_per_ms = 1 / dt
    h.dt = dt
    h.v_init = v[0]
    h.run()


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