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

    # channel to investigat
    channel = 'ka'
    gbars = ['gkabar']

    # load mechanisms
    h.nrn_load_dll('./ka/i686/.libs/libnrnmech.so')

    # parameters
    celsius = 22
    amps = [-150, -50, -150]
    durs = [150, 150, 150]
    v_steps = np.arange(-50, 100, 20)
    stepamp = 2
    pos = 0.5
    dt = 0.025

    # create cell
    cell = Section(geom={'L': 10, 'diam': 10}, Ra=100, cm=1, mechanisms={channel: {}})
    sec_channel = cell(.5).ka
    sec_channel.gkabar = 0.1

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)

    # compute response to voltage steps at different temperature
    celsius = 36.0
    sec_channel.gkabar = 0.1
    i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)