from __future__ import division
import sys
import os
from model.cell_builder import *
from neuron import h
h.load_file("stdrun.hoc")  # load NEURON libraries
h("""cvode.active(0)""")  # unvariable time step in NEURON

__author__ = 'caro'


def vclamp(v, t, cell, channel_list, E_ion, plot=False):
    """
    Note:
    - only for single compartment model
    - channel models have to be changed: current output should be zero (for better voltage clamp) and new variable to
    measure current should be inserted

    :param current_dir:
    :type current_dir:
    :param data_dir:
    :type data_dir:
    :param model_dir:
    :type model_dir:
    :param mechanism_dir:
    :type mechanism_dir:
    """

    # record from channels
    channel_currents = [0] * np.size(channel_list)
    for i, channel in enumerate(channel_list):

        # insert channel
        if channel == 'ih_fast' or channel == 'ih_slow':
            Mechanism('ih2').insert_into(cell.soma)
        else:
            Mechanism(channel + str(2)).insert_into(cell.soma)

        # set gbar to 1 and set reference to current
        if channel == 'passive':
            for seg in cell.soma:
                seg.passive2.gbar = 1
            ichannel = cell.soma(.5).passive2._ref_i2
        elif channel == 'na8st':
            for seg in cell.soma:
                seg.na8st2.gbar = 1
            ichannel = cell.soma(.5).na8st2._ref_ina2
        elif channel == 'nat':
            for seg in cell.soma:
                seg.nat2.gbar = 1
            ichannel = cell.soma(.5).nat2._ref_ina2
        elif channel == 'narsg':
            for seg in cell.soma:
                seg.narsg2.gbar = 1
            ichannel = cell.soma(.5).narsg2._ref_ina2
        elif channel == 'nap':
            for seg in cell.soma:
                seg.nap2.gbar = 1
            ichannel = cell.soma(.5).nap2._ref_ina2
        elif channel == 'nav16':
            for seg in cell.soma:
                seg.nav162.gbar = 1
            ichannel = cell.soma(.5).nav162._ref_ina2
        elif channel == 'kdr':
            for seg in cell.soma:
                seg.kdr2.gbar = 1
            ichannel = cell.soma(.5).kdr2._ref_ik2
        elif channel == 'ka':
            for seg in cell.soma:
                seg.ka2.gbar = 1
            ichannel = cell.soma(.5).ka2._ref_ik2
        elif channel == 'km':
            for seg in cell.soma:
                seg.km2.gbar = 1
            ichannel = cell.soma(.5).km2._ref_ik2
        elif channel == 'kleak':
            for seg in cell.soma:
                seg.kleak2.gbar = 1
            ichannel = cell.soma(.5).kleak2._ref_i2
        elif channel == 'ih_fast':
            for seg in cell.soma:
                seg.ih2.gfastbar = 1
                seg.ih2.gslowbar = 1
            ichannel = cell.soma(.5).ih2._ref_i_fast
        elif channel == 'ih_slow':
            for seg in cell.soma:
                seg.ih2.gfastbar = 1
                seg.ih2.gslowbar = 1
            ichannel = cell.soma(.5).ih2._ref_i_slow
        elif channel == 'caHVA':
            for seg in cell.soma:
                seg.caHVA2.gbar = 1
            ichannel = cell.soma(.5).caHVA2._ref_ica2
        elif channel == 'caLVA':
            for seg in cell.soma:
                seg.caLVA2.pbar = 1
            ichannel = cell.soma(.5).caLVA2._ref_ica2
        elif channel == 'kca':
            for seg in cell.soma:
                seg.kca2.gbar = 1
            ichannel = cell.soma(.5).kca2._ref_ik2

        # record currents
        channel_currents[i] = h.Vector()
        channel_currents[i].record(ichannel)

    # set equilibrium potentials
    for eion, E in E_ion.iteritems():
        if eion == "e_pas":
            for seg in cell.soma:
                seg.passive2.e_pas = E
        elif eion == 'ehcn':
            for seg in cell.soma:
                seg.ih2.ehcn = E
        elif eion == 'kleak':
            for seg in cell.soma:
                seg.kleak2.ekleak = E
        else:
            setattr(cell.soma, eion, E)

    # insert calcium pump if calcium is present
    if h.ismembrane('ca_ion', sec=cell.soma):
        Mechanism('cad').insert_into(cell.soma)

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
        channel_currents[i] = np.array(channel_currents[i])

        # plot current traces
        if plot:
            pl.plot(t, channel_currents[i], 'k')
            pl.ylabel('Current (mA/cm2)')
            pl.xlabel('Time (ms)')
            pl.title(channel)
            pl.show()

    return channel_currents