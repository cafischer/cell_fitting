import numpy as np
import matplotlib.pyplot as pl
from nrn_wrapper import vclamp
from optimization.helpers import get_channel_list, get_ionlist
import copy
from nrn_wrapper import iclamp

__author__ = 'caro'


def extract_simulation_params(data, sec=('soma', None), celsius=35, pos_i=0.5, pos_v=0.5, onset=200):
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
    tstop = data.t.values[-1]
    dt = data.t.values[1] - data.t.values[0]
    v_init = data.v.values[0]
    i_inj = data.i.values
    return {'i_inj': i_inj, 'v_init': v_init, 'tstop': tstop, 'dt': dt, 'pos_i': pos_i,
                                  'pos_v': pos_v, 'sec': sec, 'celsius': celsius, 'onset': onset}


def currents_given_v(v, t, sec, channel_list, ion_list, celsius, plot=False):
    """
    Records currents from sec elicited by clamping v.
    :param v: Voltage to clamp.
    :type v: array_like
    :param t: Time corresponding to v.
    :type t: array_like
    :param sec: Section where to apply the voltage clamp.
    :type sec: Section
    :param channel_list: List of ion vclamp that shall be measured.
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


def simulate_currents(cell, simulation_params, plot=False):
    channel_list = get_channel_list(cell, 'soma')
    ion_list = get_ionlist(channel_list)

    # record currents
    currents = np.zeros(len(channel_list), dtype=object)
    for i in range(len(channel_list)):
        currents[i] = cell.soma.record_from(channel_list[i], 'i' + ion_list[i], pos=.5)

    # apply vclamp
    v_model, t, i_inj = iclamp_handling_onset(cell, **simulation_params)

    # convert current traces to array
    for i in range(len(channel_list)):
        if 'onset' in simulation_params:
            real_start = int(round(simulation_params['onset'] / simulation_params['dt']))
            currents[i] = np.array(currents[i])[real_start:]
        currents[i] = np.array(currents[i])

    # plot current traces
    if plot:
        pl.figure()
        print channel_list
        for i in range(len(channel_list)):
            pl.plot(t, -1 * currents[i], label=channel_list[i])
            pl.ylabel('Current (mA/cm2)', fontsize=16)
            pl.xlabel('Time (ms)', fontsize=16)
            pl.legend(fontsize=16)
        pl.show()

    return currents


def simulate_gates(cell, simulation_params, plot=False):
    channel_list = get_channel_list(cell, 'soma')
    ion_list = get_ionlist(channel_list)

    # record gates
    gates = {}
    for ion_channel in channel_list:
        gate_names = []
        for gate_name in ['m', 'n', 'h', 'l']:
            if getattr(getattr(cell.soma(.5), ion_channel), gate_name, None) is not None:
                gate_names.append(gate_name)
        for gate_name in gate_names:
            gates[ion_channel+'_'+gate_name] = cell.soma.record_from(ion_channel, gate_name)

    # apply vclamp
    v_model, t, i_inj = iclamp_handling_onset(cell, **simulation_params)

    # convert current traces to array
    for k in gates.keys():
        if 'onset' in simulation_params:
            real_start = int(round(simulation_params['onset'] / simulation_params['dt']))
            gates[k] = np.array(gates[k])[real_start:]
        gates[k] = np.array(gates[k])

    # plot current traces
    if plot:
        pl.figure()
        for k in gates.keys():
            pl.plot(t, gates[k], label=k)
            pl.ylabel('Gate', fontsize=16)
            pl.xlabel('Time (ms)', fontsize=16)
            pl.legend(fontsize=16)
        pl.show()

    return gates


def iclamp_handling_onset(cell, **simulation_params):
    if 'onset' in simulation_params:
        onset = simulation_params['onset']
        simulation_params_tmp = copy.copy(simulation_params)
        del simulation_params_tmp['onset']
        simulation_params_tmp['tstop'] += onset
        len_onset_idx = int(round(onset / simulation_params_tmp['dt']))
        simulation_params_tmp['i_inj'] = np.concatenate((np.zeros(len_onset_idx), simulation_params_tmp['i_inj']))

        v_candidate, t_candidate = iclamp(cell, **simulation_params_tmp)

        real_start = int(round(onset / simulation_params['dt']))
        return v_candidate[real_start:], t_candidate[:-real_start], simulation_params['i_inj']
    else:
        v_candidate, t_candidate = iclamp(cell, **simulation_params)
        return v_candidate, t_candidate, simulation_params['i_inj']