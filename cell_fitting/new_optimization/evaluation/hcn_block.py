import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import os
from new_optimization.fitter import iclamp_handling_onset
from nrn_wrapper import Cell


def IV(cell, step_amp, v_init=-75):

    dt = 0.01
    step_st_ms = 200  # ms
    step_end_ms = 800  # ms
    tstop = 1000  # ms

    step_st = int(round(step_st_ms / dt))
    step_end = int(round(step_end_ms / dt)) + 1

    t_exp = np.arange(0, tstop + dt, dt)
    i_exp = np.zeros(len(t_exp))
    i_exp[step_st:step_end] = np.ones(step_end-step_st) * step_amp

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init, 'tstop': t_exp[-1],
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    return v, t

if __name__ == '__main__':
    # parameters
    #save_dir = '../../results/server/2017-07-06_13:50:52/434/L-BFGS-B/'
    #model_dir = os.path.join(save_dir, 'model', 'cell.json')
    save_dir = '../../results/hand_tuning/cell_2017-07-24_13:59:54_21_0'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    step_amp = -0.1
    data_dir = '../../data/2015_08_26b/vrest-75/IV/'+str(step_amp)+'(nA).csv'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # blocking
    cell.soma(.5).hcn_slow.gbar = 0

    # simulation
    v, t = IV(cell, step_amp, v_init=-75)
    data = pd.read_csv(data_dir)

    # plot
    save_img = os.path.join(save_dir, 'img', 'IV_blockHCN')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    pl.figure()
    pl.plot(data.t, data.v, 'k', label='Exp. Data')
    pl.plot(t, v, 'r', label='Model')
    pl.xlabel('Time $(ms)$', fontsize=16)
    pl.ylabel('Membrane potential $(mV)$', fontsize=16)
    pl.legend(loc='upper right', fontsize=16)
    pl.savefig(os.path.join(save_img, str(step_amp)+'(nA).svg'))
    pl.show()