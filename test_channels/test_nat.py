from __future__ import division
from neuron import h
from model.cell_builder import *
from test_ionchannel import *

# load NEURON libraries
h.load_file("stdrun.hoc")

# unvariable time step in NEURON
h("""cvode.active(0)""")

__author__ = 'caro'

if __name__ == "__main__":

    # channel to investigate
    channels = ["narsg", "hh2", "nat", "nafast", "nax"]
    gbars_init = [0.01, 100, 0.005, 0.003]
    gbars = ['gbar']

    # load mechanisms
    h.nrn_load_dll('./nat/i686/.libs/libnrnmech.so')

    # parameters
    celsius = 22
    amps = [-56, -100, -56]
    durs = [10, 15, 10]
    v_steps = np.arange(-100, 0, 10)
    stepamp = 2
    pos = 0.5
    dt = 0.025

    # compute response to voltage steps
    for i, channel in enumerate(channels):

        # create cell
        cell = Section(geom={'L': 10, 'diam': 10}, Ra=100, cm=1, mechanisms={channel: {'gbar': gbars_init[i]}})

        if channel =='hh2': sec_channel = cell(.5).hh2
        elif channel =='nafast': sec_channel = cell(.5).nafast
        elif channel =='nax': sec_channel = cell(.5).nax
        elif channel =='nat': sec_channel = cell(.5).nat
        elif channel =='narsg': sec_channel = cell(.5).narsg

        i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
        plot_i_steps(i_steps, v_steps, t)

    # test difference in AP shape
    v = np.zeros(len(channels), dtype=object)
    for i, channel in enumerate(channels):

        # create cell
        cell = Section(geom={'L': 10, 'diam': 10}, Ra=100, cm=1,
                       mechanisms={channel: {'gbar': gbars_init[i]}, 'hh': {'gnabar': 0.0}, 'pas': {}})
        v[i], t = spike(cell, dur=0.5, delay=2, amp=1, tstop=6, dt=0.025, v_init=-65)

    pl.figure()
    for i, channel in enumerate(channels):
        pl.plot(t, v[i], label=channel)
    pl.legend(loc='upper right')
    pl.show()

    # compute response to voltage steps at different temperature
    celsius = 36.0
    cell = Section(geom={'L': 10, 'diam': 10}, Ra=100, cm=1, mechanisms={'nat': {'gbar': 100}})
    sec_channel = cell(.5).na

    i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)