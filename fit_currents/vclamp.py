from __future__ import division
import sys
import os
from model.cell_builder import *
from neuron import h
from currentfitting import set_Epotential
h.load_file("stdrun.hoc")  # load NEURON libraries
h("""cvode.active(0)""")  # unvariable time step in NEURON

__author__ = 'caro'

# TODO: kca, cad


def vclamp(v, t, cell, channel_list, ion_list, E_ion, C_ion=None, plot=False):
    """
    Note:
    - only for single compartment model
    - channel models have to be changed:
      - current output should be zero (for better voltage clamp): delete WRITE i...
      - add "_vclamp" to the suffix of the channel
      - conductances needs to be named gbar
      - cai, cao need to be read from ca_ion
    """

    for i, channel in enumerate(channel_list):
        # insert channel and set conductance to 1
        cell.update_attr(['soma', 'mechanisms', channel+'_vclamp', 'gbar'], 1)

    # Ca concentration
    if h.ismembrane(str('ca_ion'), sec=cell.soma):
        cell.update_attr(['ion', 'ca_ion', 'cai0'], C_ion['cai'])
        cell.update_attr(['ion', 'ca_ion', 'cao0'], C_ion['cao'])

    # set equilibrium potentials
    for eion, E in E_ion.iteritems():
        if eion == "epas":
            if hasattr(cell.soma(.5), 'passive_vclamp'):
                cell.update_attr(['soma', 'mechanisms', 'passive_vclamp', eion], E)
        elif eion == 'ehcn':
            if hasattr(cell.soma(.5), 'ih_slow_vclamp'):
                cell.update_attr(['soma', 'mechanisms', 'ih_slow_vclamp', eion], E)
            if hasattr(cell.soma(.5), 'ih_fast_vclamp'):
                cell.update_attr(['soma', 'mechanisms', 'ih_fast_vclamp', eion], E)
        elif eion == 'ekleak':
            if hasattr(cell.soma(.5), 'kleak_vclamp'):
                cell.update_attr(['soma', 'mechanisms', 'kleak_vclamp', eion], E)
        else:
            cell.update_attr(['ion', eion[1:]+'_ion', eion], E)

    # record currents (do not recreate cell after pointer for record are set)
    currents = [cell.soma.record_current(channel_list[i]+'_vclamp', ion_list[i]+'_vclamp', 0.5)
                for i in range(len(channel_list))]

    # create SEClamp
    dt = t[1] - t[0]
    v_clamp = h.Vector()
    v_clamp.from_python(v)
    t_clamp = h.Vector()
    t_clamp.from_python(np.concatenate((np.array([0]), t)))  # shifted by one time step because membrane potential lags behind voltage clamp
    clamp = h.SEClamp(0.5, sec=cell.soma)
    clamp.rs = sys.float_info.epsilon  # series resistance should be as small as possible
    clamp.dur1 = 1e9
    v_clamp.play(clamp._ref_amp1, t_clamp)

    # simulate
    h.tstop = t[-1]
    h.steps_per_ms = 1 / dt
    h.dt = dt
    h.v_init = v_clamp[0]
    h.run()

    # convert current traces to array
    for i, channel in enumerate(channel_list):
        currents[i] = np.array(currents[i])

        # plot current traces
        if plot:
            pl.plot(t, currents[i], 'k')
            pl.ylabel('Current (mA/cm2)')
            pl.xlabel('Time (ms)')
            pl.title(channel)
            pl.show()

    return np.array(currents)

def vclamp_withcurrents(v, t, cell, channel_list, ion_list, E_ion, C_ion=None, plot=False):
    """
    Note:
    - only for single compartment model
    - channel models have to be changed:
      - current output should be zero (for better voltage clamp): delete WRITE i...
      - add "_vclamp" to the suffix of the channel
      - conductances needs to be named gbar
      - cai, cao need to be read from ca_ion
    """

    for i, channel in enumerate(channel_list):
        # insert channel and set conductance to 1
        cell.update_attr(['soma', 'mechanisms', channel, 'gbar'], 1)

    # Ca concentration
    if h.ismembrane(str('ca_ion'), sec=cell.soma):
        cell.update_attr(['ion', 'ca_ion', 'cai0'], C_ion['cai'])
        cell.update_attr(['ion', 'ca_ion', 'cao0'], C_ion['cao'])

    # set equilibrium potentials
    cell = set_Epotential(cell, E_ion)

    # record currents (do not recreate cell after pointer for record are set)
    currents = [cell.soma.record_current(channel_list[i], ion_list[i], 0.5) for i in range(len(channel_list))]

    # create SEClamp
    dt = t[1] - t[0]
    v_clamp = h.Vector()
    v_clamp.from_python(v)
    t_clamp = h.Vector()
    t_clamp.from_python(np.concatenate((np.array([0]), t)))  # shifted because membrane potential lags behind vclamp
    clamp = h.SEClamp(0.5, sec=cell.soma)
    clamp.rs = sys.float_info.epsilon  # series resistance should be as small as possible
    clamp.dur1 = 1e9
    v_clamp.play(clamp._ref_amp1, t_clamp)

    # simulate
    h.tstop = t[-1]
    h.steps_per_ms = 1 / dt
    h.dt = dt
    h.v_init = v_clamp[0]
    h.run()

    # convert current traces to array
    for i, channel in enumerate(channel_list):
        currents[i] = np.array(currents[i])

        # plot current traces
        if plot:
            pl.plot(t, currents[i], 'k')
            pl.ylabel('Current (mA/cm2)')
            pl.xlabel('Time (ms)')
            pl.title(channel)
            pl.show()

    return np.array(currents)


def get_ionlist(channel_list):
    ion_list = []
    for channel in channel_list:
        if 'na' in channel:
            ion_list.append('na')
        elif 'kleak' in channel:
            ion_list.append('')
        elif 'k' in channel:
            ion_list.append('k')
        elif 'ca' in channel:
            ion_list.append('ca')
        elif 'ih' in channel:
            ion_list.append('')
        else:
            ion_list.append('')
    return ion_list