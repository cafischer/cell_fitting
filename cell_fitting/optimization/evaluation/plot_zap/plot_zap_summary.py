from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import seaborn as sns
from cell_fitting.optimization.evaluation.plot_zap import simulate_and_compute_zap_characteristics
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.read_heka import get_i_inj_standard_params

pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    model_ids = range(1, 7)
    save_dir_data = '../../../data/plots/Zap20/rat/summary'
    mechanism_dir = '../../../model/channels/vavoulis'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    cell_id = '2015_08_26b'
    zap_params = get_i_inj_standard_params('Zap20')
    # zap_params['dt'] = 0.01

    load_mechanism_dir(mechanism_dir)

    # apply stim
    res_freqs_model = np.zeros(len(model_ids))
    q_values_model = np.zeros(len(model_ids))
    for i, model_id in enumerate(model_ids):
        # load model
        save_dir_model = os.path.join(save_dir, str(model_id))
        cell = Cell.from_modeldir(os.path.join(save_dir_model, 'cell.json'))

        zap_params['amp'] = 0.1 if model_id != 4 else 0.06  # model 4 spikes when amp too high
        _, _, _, _, _, res_freqs_model[i], q_values_model[i] = simulate_and_compute_zap_characteristics(cell,
                                                                                                        zap_params)

    # read data
    res_freqs_data = np.load(os.path.join(save_dir_data, 'res_freqs.npy'))
    q_values_data = np.load(os.path.join(save_dir_data, 'q_values.npy'))

    # save
    save_dir_img = os.path.join(save_dir, 'img', 'Zap20', 'res_freq_q_value')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # plot
    data = pd.DataFrame(np.array([res_freqs_data, q_values_data]).T, columns=['Res. Freq.', 'Q-Value'])
    jp = sns.jointplot('Res. Freq.', 'Q-Value', data=data, stat_func=None, color='k', alpha=0.5)
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = res_freqs_model
    jp.y = q_values_model
    jp.plot_joint(pl.scatter, c='r', alpha=0.5)
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(res_freqs_model[i]+0.05, q_values_model[i]+0.05), color='r', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'res_freq_q_value_hist.png'))
    pl.show()