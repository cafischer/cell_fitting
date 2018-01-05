import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.rampIV import simulate_rampIV
from cell_fitting.optimization.evaluation.effect_of_temperature import set_q10, set_q10_g
pl.style.use('paper')

__author__ = 'caro'


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis_with_temp'
    ramp_amp = 3.0
    q10s = [1.0, 2.0, 3.0]
    q10_g = 1.1
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    v_list = []
    for q10 in q10s:
        set_q10(cell, q10)
        set_q10_g(cell, q10_g)
        v, t, _ = simulate_rampIV(cell, ramp_amp, v_init=-75, celsius=22)
        v_list.append(v)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'effect_of_temperature', 'rampIV', 'q10_g_'+str(q10_g))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    fig, ax = pl.subplots(len(q10s), 1, sharex=True, sharey=True, figsize=(5, 8))
    for i, v in enumerate(v_list):
        ax[i].plot(t, v, 'r', label='q10=%i' % q10s[i])
        ax[i].legend()
    pl.xlabel('Time (ms)')
    fig.text(0.02, 0.5, 'Membrane Potential (mV)', va='center', rotation='vertical', fontsize=18)
    pl.subplots_adjust(top=0.96, right=0.97, left=0.17, bottom=0.08)
    pl.savefig(os.path.join(save_dir_img, 'rampIV_' + str(np.round(ramp_amp, 2)) + 'nA' + '.png'))
    pl.show()