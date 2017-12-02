from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
import os
from cell_fitting.optimization.evaluation.sine_stimulus import get_sine_stimulus
from cell_fitting.optimization.simulate import iclamp_handling_onset
from neuron import h
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    clamp_dir = '../../../model/OU_process'

    load_mechanism_dir(clamp_dir)

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # apply stim
    amp1 = 0.2  # 0.5
    amp2 = 0.4  # 0.2
    sine1_dur = 10000  # 1000 # 2000  # 5000  # 10000
    freq2 = 5  # 5  # 20
    onset_dur = 500
    offset_dur = 500
    dt = 0.01
    sine_params = {'amp1': amp1, 'amp2': amp2, 'sine1_dur': sine1_dur, 'freq2': freq2, 'onset_dur': onset_dur,
                   'offset_dur': offset_dur, 'dt': dt}

    i_inj = get_sine_stimulus(amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)

    # get simulation parameters
    # simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': -75, 'tstop': sine1_dur+onset_dur+offset_dur,
    #                      'dt': dt, 'celsius': 35, 'onset': 200}

    # conductance clamp
    a = 0.01
    conductance_clamp = h.ConductanceClamp(0.5, sec=cell.soma)
    conductance_clamp.e = 0  # TODO: E for excitatory synapses?
    #conductance_clamp.g = 1
    g_vec = h.Vector()
    g_vec.from_python(a * -1 * i_inj)  # TODO: why -1
    t_vec = h.Vector()
    t_vec.from_python(np.arange(len(i_inj))* dt)
    g_vec.play(conductance_clamp._ref_g, t_vec)  # play current into IClamp (use experimental current trace)

    # record v
    v = cell.soma.record('v', .5)
    t = h.Vector()
    t.record(h._ref_t)
    i_cc = h.Vector()
    i_cc.record(conductance_clamp._ref_i)

    h.celsius = 35
    h.v_init = -75
    h.tstop = sine1_dur+onset_dur+offset_dur
    h.steps_per_ms = 1 / dt  # change steps_per_ms before dt, otherwise dt not changed properly
    h.dt = dt
    h.run()

    v = np.array(v)
    t = np.array(t)

    # plot
    # save_dir_img = os.path.join(save_dir, 'img', 'sine_stimulus', 'doublet_test',
    #                             str(amp1)+'_'+str(amp2)+'_'+str(sine1_dur)+'_'+str(freq2))
    # if not os.path.exists(save_dir_img):
    #     os.makedirs(save_dir_img)

    pl.figure()
    pl.plot(t, i_cc, 'r')
    pl.show()

    pl.figure()
    pl.title('amp1: ' + str(amp1) + ', amp2: ' + str(amp2) + ', sine1dur: ' + str(sine1_dur) + ', freq2: ' + str(freq2), fontsize=16)
    pl.plot(t, v, 'r')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    #pl.xlim(2800, 3200)
    pl.tight_layout()
    #pl.savefig(os.path.join(save_dir_img, 'v.png'))
    pl.show()