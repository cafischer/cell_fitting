import os
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import seaborn as sns
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function
from cell_fitting.data.plot_IV.plot_fI_curve_fit_distribution import fit_fun
from cell_fitting.optimization.evaluation.plot_IV import simulate_and_compute_fI_curve, fit_fI_curve
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'
    mechanism_dir = '../../../model/channels/vavoulis'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    cell_id = '2015_08_26b'
    save_dir_data = '../../../data/plots/IV/fi_curve/rat/summary'
    model_ids = range(1, 7)
    load_mechanism_dir(mechanism_dir)

    FI_a_models = np.zeros(len(model_ids))
    FI_b_models = np.zeros(len(model_ids))
    FI_c_models = np.zeros(len(model_ids))
    RMSE = np.zeros(len(model_ids))

    for model_idx, model_id in enumerate(model_ids):
        save_dir_model = save_dir + str(model_id)

        # load model
        model_dir = os.path.join(save_dir_model, 'cell.json')
        cell = Cell.from_modeldir(model_dir)

        # fI-curve for data
        protocol = 'IV'
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         sweep_idxs=None, return_sweep_idxs=True)
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t_mat[0][-1], t_mat[0][1] - t_mat[0][0])

        amps_greater0, firing_rates_model = simulate_and_compute_fI_curve(cell)
        FI_a_models[model_idx], FI_b_models[model_idx], \
        FI_c_models[model_idx], RMSE[model_idx] = fit_fI_curve(amps_greater0, firing_rates_model)

        # plot
        save_dir_img_model = os.path.join(save_dir_model, 'img', 'plot_IV', 'fi_curve')
        if not os.path.exists(save_dir_img_model):
            os.makedirs(save_dir_img_model)

        pl.figure()
        pl.plot(amps_greater0, firing_rates_model, '-or')
        pl.plot(amps_greater0,
                fit_fun(amps_greater0, FI_a_models[model_idx], FI_b_models[model_idx], FI_c_models[model_idx]), 'b')
        pl.xlabel('Current (nA)')
        pl.ylabel('Firing rate (APs/ms)')
        pl.ylim(0, 100)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img_model, 'fIcurve_fit.png'))
        pl.show()

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'IV', 'fi_curve')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    print RMSE
    FI_a = list(np.load(os.path.join(save_dir_data, 'FI_a.npy')))
    FI_b = list(np.load(os.path.join(save_dir_data, 'FI_b.npy')))
    FI_c = list(np.load(os.path.join(save_dir_data, 'FI_c.npy')))

    data = pd.DataFrame(np.array([FI_a, FI_b]).T, columns=['Scaling', 'Shift'])
    jp = sns.jointplot('Scaling', 'Shift', data=data, stat_func=None, color='k', alpha=0.5)
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = FI_a_models
    jp.y = FI_b_models
    jp.plot_joint(pl.scatter, c='r', alpha=0.5)
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(FI_a_models[i]+3, FI_b_models[i]+0.015), color='r', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'scaling_shift_hist.png'))

    data = pd.DataFrame(np.array([FI_a, FI_c]).T, columns=['Scaling', 'Exponent'])
    jp = sns.jointplot('Scaling', 'Exponent', data=data, stat_func=None, color='k', alpha=0.5)
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = FI_a_models
    jp.y = FI_c_models
    jp.plot_joint(pl.scatter, c='r', alpha=0.5)
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(FI_a_models[i]+3, FI_c_models[i]+0.025), color='r', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'scaling_exponent_hist.png'))

    data = pd.DataFrame(np.array([FI_c, FI_b]).T, columns=['Exponent', 'Shift'])
    jp = sns.jointplot('Exponent', 'Shift', data=data, stat_func=None, color='k', alpha=0.5)
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = FI_c_models
    jp.y = FI_b_models
    jp.plot_joint(pl.scatter, c='r', alpha=0.5)
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(FI_c_models[i]+0.025, FI_b_models[i]+0.015), color='r', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'exponent_shift_hist.png'))
    pl.show()