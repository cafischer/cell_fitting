from __future__ import division

import os

import matplotlib.pyplot as pl
import numpy as np
from neuron import h
from nrn_wrapper import Cell

from cell_fitting.optimization.evaluation.plot_double_ramp.plot_doubleramp import double_ramp

pl.style.use('paper')

__author__ = 'caro'


class ChangeVmEvent(object):
    def __init__(self, section, t_event, v_new):
        self.section = section
        self.t_event = t_event
        self.v_new = v_new
        self.fih = h.FInitializeHandler(1, self.start_event)

    def start_event(self):
        h.cvode.event(self.t_event, self.set_v)

    def set_v(self):
        self.section(.5).v = self.v_new

        if (h.cvode.active()):
            h.cvode.re_init()
        else:
            h.fcurrent()




if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/1'
    model_dir = os.path.join(save_dir, 'cell.json')
    #save_dir = '../../results/server/2017-07-27_09:18:59/22/L-BFGS-B/'
    #model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../results/hand_tuning/cell_2017-07-24_13:59:54_21_0/'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # make Event
    onset = 300
    ramp_idx, t_event, v_new = 1,  493.5, -55.316  # peak of 2nd ramp, v of third ramp
    #ramp_idx, t_event, v_new = 2, 495.5, -55.322  # peak of 3rd ramp, v of 2nd ramp
    #ramp_idx, t_event, v_new = 2, 495.5, -55.47348465
    #ramp_idx, t_event, v_new = 1, 493.5, -54
    t_event += onset

    step_amp = 0
    ramp3_amp = 1.0

    # simulate normal
    t, vs, i_inj, ramp3_times, currents, channel_list = double_ramp(cell, ramp3_amp, step_amp)

    # simulate with event
    e = ChangeVmEvent(cell.soma, t_event, v_new)

    t_e, vs_e, i_inj, ramp3_times, currents, channel_list = double_ramp(cell, ramp3_amp, step_amp)

    # plot
    pl.figure()
    for v in vs[:ramp_idx]:
        pl.plot(t, v, c='r')
    for v in vs[ramp_idx+1:]:
        pl.plot(t, v, c='r')
    pl.plot(t, vs_e[ramp_idx], c='r')
    print t[np.isclose(t, t_event - onset, atol=1e-5)]
    print vs_e[ramp_idx][np.isclose(t, t_event - onset, atol=1e-5)]
    pl.plot(t_event-onset, v_new, 'ob', markersize=5, label='Setpoint of $V_m$')
    pl.xlim(490, 515)
    pl.ylim(-80, 40)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.tight_layout()
    pl.show()