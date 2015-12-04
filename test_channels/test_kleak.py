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
    channel = "kleak"
    gbars = ['g']

    # load mechanisms
    h.nrn_load_dll('./kleak/i686/.libs/libnrnmech.so')

    # create cell
    cell = Section(geom={'L': 10, 'diam': 10}, Ra=100, cm=1, mechanisms={channel: {}})
    sec_channel = cell(.5).kleak

    # parameters
    celsius = 22
    amps = [-60, -100, -60]
    durs = [1000, 3000, 1000]
    v_steps = np.arange(-100, -40, 10)
    stepamp = 2
    pos = 0.5
    dt = 0.025

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)