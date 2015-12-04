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
    channel = 'nap'
    gbars = ['gnapbar']

    # load mechanisms
    h.nrn_load_dll('./nap/i686/.libs/libnrnmech.so')

    # parameters
    celsius = 22
    amps = [-80, -60, -80]
    durs = [50, 500, 50]
    v_steps = np.arange(-60, -30, 5)
    stepamp = 2
    pos = 0.5
    dt = 0.025

    # create cell
    cell = Section(geom={'L': 10, 'diam': 10}, Ra=100, cm=1, mechanisms={"nap": {}})
    sec_channel = cell(.5).nap

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)

    # compute response to voltage steps at different temperature
    celsius = 36
    cell = Section(geom={'L': 10, 'diam': 10}, Ra=100, cm=1, mechanisms={"nap": {}})
    sec_channel = cell(.5).nap
    i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)