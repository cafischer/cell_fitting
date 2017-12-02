import os
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import seaborn as sns
from cell_characteristics.fIcurve import compute_fIcurve
from nrn_wrapper import Cell, load_mechanism_dir
from scipy.optimize import curve_fit
from cell_fitting.optimization.fitter import extract_simulation_params
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.util import merge_dicts
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function
from cell_fitting.data.IV.plot_fI_curve_fit_distribution import fit_fun
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

    FI_a_models = []
    FI_b_models = []
    FI_c_models = []
    RMSE = []

    for model_id in model_ids:
        save_dir_model = save_dir + str(model_id)

        # load model
        model_dir = os.path.join(save_dir_model, 'cell.json')
        cell = Cell.from_modeldir(model_dir)

        # fI-curve for data
        protocol = 'IV'
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         sweep_idxs=None, return_sweep_idxs=True)
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t_mat[0][-1], t_mat[0][1] - t_mat[0][0])

        # fI-curve for model
        v_mat_model = list()
        for i in range(len(sweep_idxs)):
            sim_params = {'celsius': 35, 'onset': 200}
            simulation_params = merge_dicts(extract_simulation_params(v_mat[i], t_mat[i], i_inj_mat[i]), sim_params)
            v_model, t_model, _ = iclamp_handling_onset(cell, **simulation_params)
            v_mat_model.append(v_model)

        amps, firing_rates_model = compute_fIcurve(v_mat_model, i_inj_mat, t_mat[0])

        # sort according to amplitudes
        idx_sort = np.argsort(amps)
        amps = amps[idx_sort]
        firing_rates_model = firing_rates_model[idx_sort]
        v_traces_model = np.array(v_mat_model)[idx_sort]

        # only take amps >= 0
        amps_greater0_idx = amps >= 0
        amps_greater0 = amps[amps_greater0_idx]
        firing_rates_model = firing_rates_model[amps_greater0_idx]

        # fit square root to FI-curve
        b0 = amps_greater0[np.where(firing_rates_model > 0)[0][0]]
        p_opt, _ = curve_fit(fit_fun, amps_greater0, firing_rates_model, p0=[50, b0, 0.5])
        print p_opt
        FI_a_models.append(p_opt[0])
        FI_b_models.append(p_opt[1])
        FI_c_models.append(p_opt[2])
        RMSE.append(np.sqrt(np.sum((firing_rates_model - fit_fun(amps_greater0, p_opt[0], p_opt[1], p_opt[2]))**2)))

        # plot
        save_dir_img_model = os.path.join(save_dir_model, 'img', 'IV', 'fi_curve')
        if not os.path.exists(save_dir_img_model):
            os.makedirs(save_dir_img_model)

        pl.figure()
        pl.plot(amps_greater0, firing_rates_model, '-or', label='Model')
        pl.plot(amps_greater0, fit_fun(amps_greater0, p_opt[0], p_opt[1], p_opt[2]), 'b')
        pl.xlabel('Current (nA)')
        pl.ylabel('Firing rate (APs/ms)')
        # pl.legend(loc='lower right')
        pl.ylim(0, 100)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img_model, 'fIcurve_fit.png'))
        #pl.show()

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'IV', 'fi_curve')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    print RMSE
    FI_a = list(np.load(os.path.join(save_dir_data, 'FI_a.npy')))
    FI_b = list(np.load(os.path.join(save_dir_data, 'FI_b.npy')))
    FI_c = list(np.load(os.path.join(save_dir_data, 'FI_c.npy')))

    data = pd.DataFrame(np.array([FI_a, FI_b]).T, columns=['Scaling', 'Shift'])
    jp = sns.jointplot('Scaling', 'Shift', data=data, stat_func=None, color='k')
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = FI_a_models
    jp.y = FI_b_models
    jp.plot_joint(pl.scatter, c='r')
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(FI_a_models[i]+3, FI_b_models[i]+0.015), color='r', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'scaling_shift_hist.png'))

    data = pd.DataFrame(np.array([FI_a, FI_c]).T, columns=['Scaling', 'Exponent'])
    jp = sns.jointplot('Scaling', 'Exponent', data=data, stat_func=None, color='k')
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = FI_a_models
    jp.y = FI_c_models
    jp.plot_joint(pl.scatter, c='r')
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(FI_a_models[i]+3, FI_c_models[i]+0.025), color='r', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'scaling_exponent_hist.png'))

    data = pd.DataFrame(np.array([FI_c, FI_b]).T, columns=['Exponent', 'Shift'])
    jp = sns.jointplot('Exponent', 'Shift', data=data, stat_func=None, color='k')
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = FI_c_models
    jp.y = FI_b_models
    jp.plot_joint(pl.scatter, c='r')
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(FI_c_models[i]+0.025, FI_b_models[i]+0.015), color='r', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'exponent_shift_hist.png'))
    pl.show()