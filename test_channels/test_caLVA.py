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
    channels = ['itGHK']  # ['CAtM95', 'cal', 'cat', 'ical']
    gbars = ['gbar']

    # load mechanisms
    h.nrn_load_dll('./caLVA/i686/.libs/libnrnmech.so')

    # parameters
    celsius = 20
    amps = [-135, -70, -135]
    durs = [500, 200, 500]
    v_steps = np.arange(-70, -30, 10)
    stepamp = 2
    pos = 0.5
    dt = 0.025

for i, channel in enumerate(channels):

        # create cell
        cell = Section(geom={'L': 10, 'diam': 10}, Ra=100, cm=1, mechanisms={channel: {}})

        if channel == 'cal': sec_channel = cell(.5).cal
        elif channel == 'cat': sec_channel = cell(.5).cat
        elif channel == 'ical': sec_channel = cell(.5).ical
        elif channel == 'CAtM95': sec_channel = cell(.5).CAtM95
        elif channel == 'itGHK': sec_channel = cell(.5).itGHK

        # compute response to voltage steps
        setattr(sec_channel, gbars[0], 0.00015)
        i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
        plot_i_steps(i_steps, v_steps, t)

        # compute response to voltage steps at different temperature
        celsius = 36
        setattr(sec_channel, gbars[0], 0.00015)
        i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
        plot_i_steps(i_steps, v_steps, t)