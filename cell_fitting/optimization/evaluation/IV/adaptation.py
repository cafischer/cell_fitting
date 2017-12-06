import os
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import seaborn as sns
from nrn_wrapper import Cell, load_mechanism_dir
from scipy.optimize import curve_fit
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.optimization.fitter import extract_simulation_params
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.util import merge_dicts, init_nan
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function, get_sweep_index_for_amp
from cell_fitting.data.IV.adaptation import get_trace_with_n_APs, fit_fun
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'
    mechanism_dir = '../../../model/channels/vavoulis'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    cell_id = '2015_08_26b'
    save_dir_data = '../../../data/plots/IV/adaptation/rat/summary'
    model_ids = range(1, 7)
    load_mechanism_dir(mechanism_dir)
    n_APs = 25
    AP_threshold = 0

    f_mins_model = init_nan(len(model_ids))
    f_maxs_model = init_nan(len(model_ids))
    taus_model = init_nan(len(model_ids))
    RMSE_model = init_nan(len(model_ids))

    for mi, model_id in enumerate(model_ids):
        save_dir_model = save_dir + str(model_id)

        # load model
        model_dir = os.path.join(save_dir_model, 'cell.json')
        cell = Cell.from_modeldir(model_dir)

        # simulate
        protocol = 'IV'
        amp_max = 2.0
        sweep_idx_max = get_sweep_index_for_amp(amp_max, protocol)
        tstop = 1000
        dt = 0.01
        i_inj_mat = get_i_inj_from_function(protocol, range(sweep_idx_max), tstop, dt)
        v_mat_model = []
        t_mat_model = []
        for i_inj in i_inj_mat:
            simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': -75, 'tstop': tstop,
                                 'dt': dt, 'celsius': 35, 'onset': 200}
            v_model, t_model, _ = iclamp_handling_onset(cell, **simulation_params)
            v_mat_model.append(v_model)
            t_mat_model.append(t_model)

        # find trace with right number of APs
        v, t = get_trace_with_n_APs(v_mat_model, t_mat_model, n_APs, AP_threshold)
        if v is None:
            print 'Model'+str(model_id)+' has not '+str(n_APs)+'!'
            continue

        # cut off on- and offset
        dt = t[1] - t[0]
        start_i_inj_idx = to_idx(250, dt)
        end_i_inj_idx = to_idx(750, dt)
        v = v[start_i_inj_idx:end_i_inj_idx]
        t = t[start_i_inj_idx:end_i_inj_idx] - t[start_i_inj_idx]

        # get instantaneous frequency
        AP_onset_idxs = get_AP_onset_idxs(v, AP_threshold)
        ISIs = np.diff(t[AP_onset_idxs])
        f_inst = 1. / ISIs * 1000  # Hz
        t_inst = t[AP_onset_idxs[:-1]]
        #f_inst = f_inst[1:]
        #t_inst = t_inst[1:]

        # fit
        try:
            p_opt, _ = curve_fit(fit_fun, t_inst, f_inst, p0=[np.min(f_inst), np.max(f_inst), 1])
        except RuntimeError:
            print 'Model' + str(model_id) + ' could not be fit!'
            continue

        f_mins_model[mi] = p_opt[0]
        f_maxs_model[mi] = p_opt[1]
        taus_model[mi] = p_opt[2]
        RMSE_model[mi] = np.sqrt(np.sum((f_inst - fit_fun(t_inst, p_opt[0], p_opt[1], p_opt[2])) ** 2))

        # plot
        save_dir_img_model = os.path.join(save_dir_model, 'img', 'IV', 'adaptation')
        if not os.path.exists(save_dir_img_model):
            os.makedirs(save_dir_img_model)

        print 'RMSE: %.5f' % RMSE_model[mi]
        print 'p_opt: ' + str(p_opt)
        pl.figure()
        pl.plot(t_inst, f_inst, '-ok', label='Exp. Data')
        pl.plot(t_inst, fit_fun(t_inst, p_opt[0], p_opt[1], p_opt[2]), 'b')
        pl.xlabel('Time (ms)')
        pl.ylabel('Instantaneous Firing Rate (Hz)')
        #pl.ylim(0, 100)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img_model, 'adaptation_fit.png'))
        pl.show()

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'IV', 'adaptation')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    f_mins_data = list(np.load(os.path.join(save_dir_data, 'f_mins.npy')))
    f_maxs_data = list(np.load(os.path.join(save_dir_data, 'f_maxs.npy')))
    taus_data = list(np.load(os.path.join(save_dir_data, 'taus.npy')))

    data = pd.DataFrame(np.array([f_mins_data, f_maxs_data]).T, columns=['f_min', 'f_max'])
    jp = sns.jointplot('f_min', 'f_max', data=data, stat_func=None, color='k')
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = f_mins_model
    jp.y = f_maxs_model
    jp.plot_joint(pl.scatter, c='r')
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(f_mins_model[i]+0.1, f_maxs_model[i]+0.1), color='r', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'f_min_f_max_hist.png'))

    data = pd.DataFrame(np.array([f_mins_data, taus_data]).T, columns=['f_min', 'tau'])
    jp = sns.jointplot('f_min', 'tau', data=data, stat_func=None, color='k')
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = f_mins_model
    jp.y = taus_model
    jp.plot_joint(pl.scatter, c='r')
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(f_mins_model[i]+0.1, taus_model[i]+0.1), color='r', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'f_min_tau_hist.png'))

    data = pd.DataFrame(np.array([taus_data, f_maxs_data]).T, columns=['tau', 'f_max'])
    jp = sns.jointplot('tau', 'f_max', data=data, stat_func=None, color='k')
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = taus_model
    jp.y = f_maxs_model
    jp.plot_joint(pl.scatter, c='r')
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(taus_model[i]+0.1, f_maxs_model[i]+0.1), color='r', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'tau_f_max_hist.png'))
    pl.show()