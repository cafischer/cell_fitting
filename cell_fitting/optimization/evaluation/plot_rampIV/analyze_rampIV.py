import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.optimization.evaluation.plot_rampIV import simulate_rampIV, find_current_threshold, plot_rampIV, \
    load_rampIV_data, get_rmse
from nrn_wrapper import Cell
import time
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    #save_dir = '/home/cf/Phd/server/cns/server/results/sensitivity_analysis/2017-10-10_14:00:01/3519'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    ramp_amp = 3.1
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    start_time = time.time()
    v, t, _ = simulate_rampIV(cell, ramp_amp, v_init=-75)
    end_time = time.time()
    print 'Runtime (sec): ', end_time - start_time

    # current to elicit AP
    current_threshold = find_current_threshold(cell)
    print 'Current threshold: %.2f nA' % current_threshold

    v_data, t_data, i_inj_data = load_rampIV_data(data_dir, ramp_amp, v_shift=-16)
    dt = t_data[1] - t_data[0]

    # rmse
    rmse, rmse_dap = get_rmse(v, v_data, t_data, i_inj_data, dt)
    print 'RMSE: %.2f mV' % rmse
    print 'RMSE DAP: %.2f mV' % rmse_dap

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'rampIV')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.savetxt(os.path.join(save_dir_img, 'current_threshold.txt'), np.array([current_threshold]))

    if not os.path.exists(os.path.join(save_dir_img, '%.2f(nA)' % ramp_amp)):
        os.makedirs(os.path.join(save_dir_img, '%.2f(nA)' % ramp_amp))

    pl.figure()
    #pl.title(str(np.round(ramp_amp, 2)) + ' nA')
    pl.plot(t_data, v_data, 'k', label='Exp. Data')
    pl.plot(t, v, 'r', label='Model')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.legend(loc='upper right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, '%.2f(nA)' % ramp_amp, 'rampIV_with_data.png'))

    plot_rampIV(t, v, os.path.join(save_dir_img, '%.2f(nA)' % ramp_amp))