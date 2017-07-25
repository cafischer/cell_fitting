from __future__ import division
import pylab as pl
import numpy as np
from matplotlib.pyplot import cm
from nrn_wrapper import Cell
from optimization.simulate import iclamp_handling_onset
import os

__author__ = 'caro'


def get_ramp(start_idx, end_idx, amp_before, ramp_amp, amp_after):
    diff_idx = end_idx - start_idx
    half_diff_up = int(np.ceil(diff_idx / 2))
    half_diff_down = int(np.floor(diff_idx / 2))
    i_exp = np.zeros(diff_idx)
    i_exp[:half_diff_up] = np.linspace(amp_before, ramp_amp, half_diff_up)
    i_exp[half_diff_up:] = np.linspace(ramp_amp, amp_after, half_diff_down+1)[1:]
    return i_exp


def double_ramp(cell, ramp3_amp, step_amp):
    """
    original values
    delta_ramp = 2
    delta_first = 3
    ramp3_times = np.arange(delta_first, 10 * delta_ramp + delta_ramp, delta_ramp)
    baseline_amp = -0.05
    ramp_amp = 4.0
    ramp3_amp = 1.8
    step_amp = 0  # or -0.1 or 0.1
    dt = 0.01

    amplitude of second ramp goes up by 0.05 nA after each sequence
    """

    delta_ramp = 2
    delta_first = 3
    ramp3_times = np.arange(delta_first, 10 * delta_ramp + delta_ramp, delta_ramp)
    baseline_amp = -0.05
    ramp_amp = 4.0
    dt = 0.01

    # construct current traces
    len_ramp = 3
    start_ramp1 = int(round(20 / dt))
    end_ramp1 = start_ramp1 + int(round(len_ramp / dt))
    start_step = int(round(222 / dt))
    end_step = start_step + int(round(250 / dt))
    start_ramp2 = end_step + int(round(15 / dt))
    end_ramp2 = start_ramp2 + int(round(len_ramp / dt))

    t_exp = np.arange(0, 800, dt)
    v = np.zeros([len(ramp3_times), len(t_exp)])

    for j, ramp3_time in enumerate(ramp3_times):
        start_ramp3 = start_ramp2 + int(round(ramp3_time / dt))
        end_ramp3 = start_ramp3 + int(round(len_ramp / dt))

        i_exp = np.ones(len(t_exp)) * baseline_amp
        i_exp[start_ramp1:end_ramp1] = get_ramp(start_ramp1, end_ramp1, 0, ramp_amp, 0)
        i_exp[start_step:end_step] = step_amp
        i_exp[start_ramp2:end_ramp2] = get_ramp(start_ramp2, end_ramp2, 0, ramp_amp, 0)
        i_exp[start_ramp3:end_ramp3] = get_ramp(start_ramp3, end_ramp3, 0, ramp3_amp, 0)

        # get simulation parameters
        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -75, 'tstop': t_exp[-1],
                             'dt': dt, 'celsius': 35, 'onset': 300}

        # record v
        v[j], t, _ = iclamp_handling_onset(cell, **simulation_params)

        # record currents
        #currents = simulate_currents(fitter.cell, simulation_params, plot=True)

    # plot
    pl.figure()
    pl.title('1st Ramp = 4 nA, 2nd Ramp = ' + str(ramp3_amp) + ' nA')
    color = iter(cm.gist_rainbow(np.linspace(0, 1, len(ramp3_times))))
    for j, ramp3_time in enumerate(ramp3_times):
        pl.plot(t, v[j], label='Model' if j == 0 else '', c='r')
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Membrane potential (mV)', fontsize=16)
    pl.xlim(485, 560)
    pl.tight_layout()
    pl.legend(fontsize=16)
    pl.savefig(save_dir_img)
    pl.show()


if __name__ == '__main__':

    # parameters
    #save_dir = '../../results/server/2017-07-17_17:05:19/54/L-BFGS-B/'
    #model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../results/hand_tuning/cell434_5/'
    #model_dir = os.path.join(save_dir, 'cell.json')
    save_dir = '../../results/hand_tuning/test0/'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    step_amp = -0.1
    for seq in range(20):
        save_dir_img = os.path.join(save_dir, 'img', 'PP', 'step'+str(step_amp), 'PP'+str(seq)+'.png')
        if not os.path.exists(os.path.join(save_dir, 'img', 'PP', 'step'+str(step_amp))):
            os.makedirs(os.path.join(save_dir, 'img', 'PP', 'step'+str(step_amp)))
        ramp3_amp = 1.0 + seq * 0.1  # TODO 1.8 + seq * 0.05
        double_ramp(cell, ramp3_amp, step_amp)  # 4, 5, 7 # 1, 4, 5  # 0, 0, 2