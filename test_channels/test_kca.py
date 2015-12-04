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
    channels = ['kca']

    # load mechanisms
    h.nrn_load_dll('./kca/i686/.libs/libnrnmech.so')
    gbars = ['gbar']

    # parameters
    celsius = 22
    amps = [-50, 20, -110]
    durs = [10, 5, 16]
    v_steps = np.arange(-110, -40, 10)
    stepamp = 3
    pos = 0.5
    dt = 0.025

for i, channel in enumerate(channels):

        # create cell
        cell = Section(geom={'L': 10, 'diam': 10}, Ra=100, cm=1, mechanisms={channel: {}})

        if channel == 'iahp': sec_channel = cell(.5).iahp
        elif channel == 'kca': sec_channel = cell(.5).kca
        elif channel == 'skkca': sec_channel = cell(.5).skkca
        elif channel == 'bk': sec_channel = cell(.5).bk

        # compute response to voltage steps
        val = 1.6
        setattr(sec_channel, gbars[0], val)
        if channel=='kca' or channel=='ba': setattr(sec_channel, gbars[0], val*10**4)

        i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
        plot_i_steps(i_steps, v_steps, t)

        # compute response to voltage steps at different temperature
        celsius = 36
        setattr(sec_channel, gbars[0], val)

        i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
        plot_i_steps(i_steps, v_steps, t)