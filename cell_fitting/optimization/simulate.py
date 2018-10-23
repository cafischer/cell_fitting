import copy
import re
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import iclamp, iclamp_adaptive, vclamp
from cell_fitting.optimization.helpers import get_channel_list, get_ionlist
__author__ = 'caro'


# def extract_simulation_params(data, sec=('soma', None), celsius=35, pos_i=0.5, pos_v=0.5, onset=200):
#     """
#     Uses the experimental data and additional arguments to extract the simulation parameters.
#
#     :param data: Dataframe containing the columns: v (membrane potential), t (time), i (injected current).
#     :type data: pandas.DataFrame
#     :param sec: Section to stimulate and record from. First argument name, second argument index (None for no index).
#     :type sec: tuple
#     :param celsius: Temperature during simulation (affects ion channel kinetics).
#     :type celsius: float
#     :param pos_i: Position of the stimulating electrode (value between 0 and 1).
#     :type pos_i: float
#     :param pos_v: Position of the recording electrode (value between 0 and 1).
#     :type pos_v: float
#     :return: Simulation parameter
#     :rtype: dict
#     """
#     tstop = data.t.values[-1]
#     dt = data.t.values[1] - data.t.values[0]
#     v_init = data.v.values[0]
#     i_inj = data.i.values
#     return {'i_inj': i_inj, 'v_init': v_init, 'tstop': tstop, 'dt': dt, 'pos_i': pos_i,
#             'pos_v': pos_v, 'sec': sec, 'celsius': celsius, 'onset': onset}


def extract_simulation_params(v, t, i_inj, sec=('soma', None), celsius=35, pos_i=0.5, pos_v=0.5, onset=200):
    """
    Uses the experimental data and additional arguments to extract the simulation parameters.

    :param v: Membrane potential.
    :type v: numpy.array
    :param t: Time.
    :type t: numpy.array
    :param i_inj: Injected current.
    :type i_inj: numpy.array
    :param sec: Section to stimulate and record from. First argument name, second argument index (None for no index).
    :type sec: tuple
    :param celsius: Temperature during simulation (affects ion channel kinetics).
    :type celsius: float
    :param pos_i: Position of the stimulating electrode (value between 0 and 1).
    :type pos_i: float
    :param pos_v: Position of the recording electrode (value between 0 and 1).
    :type pos_v: float
    :return: Simulation parameter
    :rtype: dict
    """
    tstop = t[-1]
    dt = t[1] - t[0]
    v_init = v[0]
    return {'i_inj': i_inj, 'v_init': v_init, 'tstop': tstop, 'dt': dt, 'pos_i': pos_i,
            'pos_v': pos_v, 'sec': sec, 'celsius': celsius, 'onset': onset}


def get_standard_simulation_params():
    return {'v_init': -75, 'dt': 0.01, 'pos_i': 0.5, 'pos_v': 0.5, 'sec': ('soma', None), 'celsius': 35, 'onset': 200}


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
            pl.ylabel('Current (mA/cm$^2$)', fontsize=16)
            pl.xlabel('Time (ms)', fontsize=16)
        pl.legend(fontsize=16)
        pl.tight_layout()
        pl.show()

    return currents, channel_list


def simulate_gates(cell, simulation_params, return_vh_vs=False, plot=False):
    channel_list = get_channel_list(cell, 'soma')

    # record gates
    gates = {}
    power_gates = {}
    vh_gates = {}
    vs_gates = {}
    for ion_channel in channel_list:
        gate_names = []
        for gate_name in ['m', 'n', 'h', 'l']:
            if getattr(getattr(cell.soma(.5), ion_channel), gate_name, None) is not None:
                gate_names.append(gate_name)
        for gate_name in gate_names:
            gates[ion_channel+'_'+gate_name] = cell.soma.record_from(ion_channel, gate_name)
            power_gates[ion_channel+'_'+gate_name] = cell.get_attr(['soma', '0.5', ion_channel, gate_name+'_pow'])
            if return_vh_vs:
                vh_gates[ion_channel+'_'+gate_name] = cell.get_attr(['soma', '0.5', ion_channel, gate_name+'_vh'])
                vs_gates[ion_channel+'_'+gate_name] = cell.get_attr(['soma', '0.5', ion_channel, gate_name + '_vs'])

    # apply vclamp
    v_model, t, i_inj = iclamp_handling_onset(cell, **simulation_params)

    # convert current traces to array
    for k in gates.keys():
        if 'onset' in simulation_params:
            real_start = int(round(simulation_params['onset'] / simulation_params['dt']))
            gates[k] = np.array(gates[k])[real_start:]
        gates[k] = np.array(gates[k])

    # plot gate traces
    if plot:
        new_channel_names = {k: k for k in gates.keys()}
        new_channel_names['nap_m'] = 'nat_m'
        new_channel_names['nap_h'] = 'nat_h'
        new_channel_names['nat_m'] = 'nap_m'
        new_channel_names['nat_h'] = 'nap_h'

        fig, ax1 = pl.subplots()
        for k in sorted(new_channel_names.keys(), reverse=True):
            ax1.plot(t, gates[k] ** power_gates[k], label=new_channel_names[k])
            ax1.set_ylabel('Gate')
            ax1.set_xlabel('Time (ms)')
        ax2 = ax1.twinx()
        ax2.plot(t, v_model, 'k')
        ax2.set_ylabel('Membrane Potential (mV)')
        ax2.spines['right'].set_visible(True)
        ax1.legend()
        ax2.set_ylim(-80, -40)
        pl.tight_layout()
        pl.show()

        pl.figure()
        gates_keys = gates.keys()
        for ion_channel in channel_list:
            channel_gates = np.ones(len(t))
            regexp = re.compile(ion_channel + '_.+')
            for key in gates_keys:
                if re.match(regexp, key):
                    channel_gates *= gates[key] ** power_gates[key]
            pl.plot(t, channel_gates, label=ion_channel)
        pl.legend()
        pl.show()

    if return_vh_vs:
        return gates, power_gates, vh_gates, vs_gates
    return gates, power_gates


def iclamp_handling_onset(cell, **simulation_params):
    if 'onset' in simulation_params:
        onset = simulation_params['onset']
        simulation_params_tmp = copy.copy(simulation_params)
        del simulation_params_tmp['onset']
        simulation_params_tmp['tstop'] += onset
        len_onset_idx = int(round(onset / simulation_params_tmp['dt']))
        simulation_params_tmp['i_inj'] = np.concatenate((np.ones(len_onset_idx) *  simulation_params_tmp['i_inj'][0],
                                                         simulation_params_tmp['i_inj']))

        v_candidate, t_candidate = iclamp(cell, **simulation_params_tmp)

        real_start = int(round(onset / simulation_params['dt']))
        if onset == 0:  # indexing does not work for onset = 0, therefore this extra branch
            return v_candidate, t_candidate, simulation_params['i_inj']
        else:
            return v_candidate[real_start:], t_candidate[:-real_start], simulation_params['i_inj']
    else:
        v_candidate, t_candidate = iclamp(cell, **simulation_params)
        return v_candidate, t_candidate, simulation_params['i_inj']


def iclamp_adaptive_handling_onset(cell, **simulation_params):
    if 'onset' in simulation_params:
        onset = simulation_params['onset']
        simulation_params_tmp = copy.copy(simulation_params)
        del simulation_params_tmp['onset']
        simulation_params_tmp['tstop'] += onset
        len_onset_idx = int(round(onset / simulation_params_tmp['dt']))
        simulation_params_tmp['i_inj'] = np.concatenate((np.ones(len_onset_idx) * simulation_params_tmp['i_inj'][0],
                                                         simulation_params_tmp['i_inj']))

        v_candidate, t_candidate = iclamp_adaptive(cell, **simulation_params_tmp)

        real_start = np.where(t_candidate >= onset)[0][0]
        return v_candidate[real_start:], t_candidate[real_start:] - onset, simulation_params['i_inj']
    else:
        v_candidate, t_candidate = iclamp_adaptive(cell, **simulation_params)
        return v_candidate, t_candidate, simulation_params['i_inj']