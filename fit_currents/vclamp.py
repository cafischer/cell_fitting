from __future__ import division
import sys
import os
from model.cell_builder import *
from neuron import h
h.load_file("stdrun.hoc")  # load NEURON libraries
h("""cvode.active(0)""")  # unvariable time step in NEURON

__author__ = 'caro'


def vclamp(v, t, cell, channel_list, ion_list, E_ion, C_ion=None, plot=False):
    """
    Note:
    - only for single compartment model
    - channel models have to be changed: current output should be zero (for better voltage clamp) and new variable to
    measure current should be inserted
    """

    # record from channels
    currents = [0] * np.size(channel_list)
    for i, channel in enumerate(channel_list):

        # insert channel & set gbar to 1 & record currents
        currents[i] = h.Vector()
        if channel == 'ih_fast' or channel == 'ih_slow':
            Mechanism('ih2').insert_into(cell.soma)
            currents[i].record(getattr(getattr(cell.soma(.5), 'ih2'), '_ref_i'+ion_list[i]))
            setattr(getattr(cell.soma(.5), 'ih2'), 'gfastbar', 1)
            setattr(getattr(cell.soma(.5), 'ih2'), 'gslowbar', 1)
        elif channel == 'calva':
            Mechanism(channel + str(2)).insert_into(cell.soma)
            currents[i].record(getattr(getattr(cell.soma(.5), channel+'2'), '_ref_i'+ion_list[i]+'2'))
            setattr(getattr(cell.soma(.5), channel+'2'), 'pbar', 1)
            setattr(getattr(cell.soma(.5), channel+'2'), 'cai', C_ion['cai'])
            setattr(getattr(cell.soma(.5), channel+'2'), 'cao', C_ion['cao'])
        elif channel == 'kca':
            Mechanism(channel + str(2)).insert_into(cell.soma)
            currents[i].record(getattr(getattr(cell.soma(.5), channel+'2'), '_ref_i'+ion_list[i]+'2'))
            setattr(getattr(cell.soma(.5), channel+'2'), 'gbar', 1)
            setattr(getattr(cell.soma(.5), channel+'2'), 'cai', C_ion['cai'])
        else:
            Mechanism(channel + str(2)).insert_into(cell.soma)
            currents[i].record(getattr(getattr(cell.soma(.5), channel+'2'), '_ref_i'+ion_list[i]+'2'))
            setattr(getattr(cell.soma(.5), channel+'2'), 'gbar', 1)

    # set equilibrium potentials
    for eion, E in E_ion.iteritems():
        if eion == "e_pas":
            for seg in cell.soma:
                seg.passive2.e_pas = E
        elif eion == 'ehcn':
            for seg in cell.soma:
                seg.ih2.ehcn = E
        elif eion == 'ekleak':
            for seg in cell.soma:
                seg.kleak2.ekleak = E
        else:
            setattr(cell.soma, eion, E)

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
        elif '_fast' in channel:
            ion_list.append('_fast')
        elif '_slow' in channel:
            ion_list.append('_slow')
        else:
            ion_list.append('')
    return ion_list
