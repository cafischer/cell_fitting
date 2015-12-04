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
    channels = ["narsg", "na8st", "nat"]
    gbars = ['gbar']

    # load mechanisms
    h.nrn_load_dll('./rsg_na/i686/.libs/libnrnmech.so')

    # parameters
    celsius = 22
    amps = [-90, 30, -60]
    durs = [20, 20, 100]
    v_steps = np.arange(-60, -10, 10)
    stepamp = 3
    pos = 0.5
    dt = 0.025

    for channel in channels:
        # create cell
        cell = Section(geom={'L': 10, 'diam': 10}, nseg=1, Ra=100, cm=1, mechanisms={channel: {}})
        if channel == 'narsg': sec_channel = cell(.5).narsg
        elif channel =='na8st': sec_channel = cell(.5).na8st

        # compute response to voltage steps
        i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
        plot_i_steps(i_steps, v_steps, t)

    # compute response to voltage steps at different temperature
    celsius = 36
    cell = Section(geom={'L': 10, 'diam': 10}, nseg=1, Ra=100, cm=1, mechanisms={'narsg': {}})
    sec_channel = cell(.5).narsg

    i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)