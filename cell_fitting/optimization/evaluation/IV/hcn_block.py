import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import os
from cell_fitting.optimization.fitter import iclamp_handling_onset
from nrn_wrapper import Cell
pl.style.use('paper')


def IV(cell, step_amp, v_init=-75):

    dt = 0.01
    step_st_ms = 250  # ms
    step_end_ms = 750  # ms
    tstop = 1150  # ms

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

    # simulation
    v_before, t_before = IV(cell, step_amp, v_init=-75)

    # blocking
    cell.soma(.5).hcn_slow.gbar = 0

    # simulation
    v_after, t_after = IV(cell, step_amp, v_init=-75)

    # data
    data = pd.read_csv(data_dir)

    # plot
    save_img = os.path.join(save_dir, 'img', 'IV_blockHCN')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    pl.figure()
    #pl.plot(data.t, data.v, 'k', label='Exp. Data')
    pl.plot(t_before, v_before, 'r', label='before ZD')
    pl.plot(t_after, v_after, 'r', label='after ZD', alpha=0.5)
    st = np.ceil(pl.ylim()[1] / 5) * 5
    pl.yticks(np.arange(st, st + 7 * -5, -5))
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend(loc='lower right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_img, str(step_amp)+'(nA).png'))
    pl.show()


    pl.figure()
    #pl.plot(data.t, data.v, 'k', label='Exp. Data')
    pl.plot(t_before, v_before, 'r', label='before ZD')
    pl.plot(t_after, v_after, 'r', label='after ZD', alpha=0.5)
    pl.ylim(-87, -72)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend(loc='lower right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_img, str(step_amp)+'(nA)_zoom.png'))
    pl.show()