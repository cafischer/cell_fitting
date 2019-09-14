from __future__ import division
import matplotlib.pyplot as pl
import json
import os
from nrn_wrapper import Cell
from cell_fitting.read_heka import get_i_inj_standard_params
from cell_fitting.optimization.evaluation.plot_zap import simulate_and_compute_zap_characteristics, \
    plot_v_model_and_exp, plot_v_and_impedance
pl.style.use('paper')


def plot_zap(v, t,  imp_smooth, frequencies, res_freq, q_value, zap_params, save_dir_img=None, data_dir=None):
    plot_v_model_and_exp(v, t, zap_params, save_dir_img, data_dir)

    plot_v_and_impedance(zap_params['freq0'], zap_params['freq1'], frequencies, imp_smooth,
                         zap_params['offset_dur'], zap_params['onset_dur'],
                         q_value, res_freq, save_dir_img, t, zap_params['tstop'], v, v[0])
    pl.show()


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cfischer/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    #data_dir = '/home/cfischer/Phd/DAP-Project/cell_data/raw_data'
    data_dir = '/media/cfischer/TOSHIBA EXT/Sicherung_2018_05_19/Phd/DAP-Project/cell_data/raw_data'
    cell_id = '2015_08_26b'
    zap_params = get_i_inj_standard_params('Zap20')
    #zap_params['dt'] = 0.01
    #zap_params['amp'] = 0.15

    save_dir_img = os.path.join(save_dir, 'img', 'zap')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # simulate zap and compute characteristics
    v, t, i_inj, imp_smooth, frequencies, res_freq, q_value = simulate_and_compute_zap_characteristics(cell, zap_params)

    # plot
    data_dir = os.path.join(data_dir, cell_id + '.dat')
    plot_zap(v, t, imp_smooth, frequencies, res_freq, q_value, zap_params, save_dir_img, data_dir)

    # save
    impedance_dict = dict(impedance=list(imp_smooth), frequencies=list(frequencies))
    with open(os.path.join(save_dir_img, 'impedance_dict.json'), 'w') as f:
        json.dump(impedance_dict, f)
