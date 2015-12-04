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

    # channels to investigate
    channels = ['caHVA', 'cal', 'CAlM95']
    gbars = ['gbar']

    # load mechanisms
    h.nrn_load_dll('./caHVA/i686/.libs/libnrnmech.so')

    # parameters
    celsius = 20
    amps = [-60, -40, -60]
    durs = [500, 200, 500]
    v_steps = np.arange(-40, 40, 10)
    stepamp = 2
    pos = 0.5
    dt = 0.025


for i, channel in enumerate(channels):

        # create cell
        cell = Section(geom={'L': 10, 'diam': 10}, Ra=100, cm=1, mechanisms={channel: {}})

        if channel == 'caHVA': sec_channel = cell(.5).caHVA
        elif channel == 'sca': sec_channel = cell(.5).sca
        elif channel == 'cal': sec_channel = cell(.5).cal
        elif channel == 'CAlM95': sec_channel = cell(.5).CAlM95

        # compute response to voltage steps
        setattr(sec_channel, gbars[0], 50)
        i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
        plot_i_steps(i_steps, v_steps, t)

        # compute response to voltage steps at different temperature
        celsius = 36
        setattr(sec_channel, gbars[0], 50)
        i_steps, t = current_subtraction(cell, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
        plot_i_steps(i_steps, v_steps, t)