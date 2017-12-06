from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from scipy.optimize import curve_fit
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_characteristics import to_idx
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
import seaborn as sns
import pandas as pd
pl.style.use('paper')


def fit_fun(x, f_min, f_max, tau):
    return f_min + (f_max - f_min) * np.exp(-x / tau)


def get_trace_with_n_APs(v_mat, t_mat, n_APs, AP_threshold):
    n_APs_per_trace = np.array([len(get_AP_onset_idxs(v, AP_threshold)) for v in v_mat])
    right_n_AP_idx = np.where(n_APs_per_trace == n_APs)[0]
    if len(right_n_AP_idx) == 0:
        right_n_AP_idx = np.where(np.logical_and(n_APs_per_trace >= n_APs - 1, n_APs_per_trace <= n_APs + 1))[0]
        if len(right_n_AP_idx) == 0:
            right_n_AP_idx = np.where(np.logical_and(n_APs_per_trace >= n_APs - 2, n_APs_per_trace <= n_APs + 2))[0]
            if len(right_n_AP_idx) == 0:
                 return None, None
    return v_mat[right_n_AP_idx[0]], t_mat[right_n_AP_idx[0]]


if __name__ == '__main__':

    # parameters
    save_dir_img = '../plots/IV/adaptation/rat/summary'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'IV'
    animal = 'rat'
    #cell_ids = get_cells_for_protocol(data_dir, protocol)
    #cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)
    cell_ids = np.load('../plots/IV/fi_curve/rat/summary/cells.npy')
    AP_threshold = 0
    n_APs = 25

    f_mins = []
    f_maxs = []
    taus = []
    cells = []
    RMSE = []

    for cell_id in cell_ids:
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        dt = t_mat[0, 1] - t_mat[0, 0]

        # find trace with right number of APs
        v, t = get_trace_with_n_APs(v_mat, t_mat, n_APs, AP_threshold)
        if v is None:
            continue

        # cut off on- and offset
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
            continue

        f_mins.append(p_opt[0])
        f_maxs.append(p_opt[1])
        taus.append(p_opt[2])
        cells.append(cell_id)
        RMSE.append(np.sqrt(np.sum((f_inst - fit_fun(t_inst, p_opt[0], p_opt[1], p_opt[2])) ** 2)))

        print 'RMSE: %.5f' % RMSE[-1]
        print 'p_opt: ' + str(p_opt)
        # pl.figure()
        # pl.plot(t_inst, f_inst, '-ok', label='Exp. Data')
        # pl.plot(t_inst, fit_fun(t_inst, p_opt[0], p_opt[1], p_opt[2]), 'b')
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Instantaneous Firing Rate (Hz)')
        # #pl.ylim(0, 100)
        # pl.tight_layout()
        # #pl.show()

    # plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'f_mins.npy'), f_mins)
    np.save(os.path.join(save_dir_img, 'f_maxs.npy'), f_maxs)
    np.save(os.path.join(save_dir_img, 'taus.npy'), taus)
    np.save(os.path.join(save_dir_img, 'RMSE.npy'), RMSE)
    np.save(os.path.join(save_dir_img, 'cells.npy'), cells)

    pl.figure()
    pl.hist(f_mins, bins=100, color='0.5')
    pl.ylabel('Count')
    pl.xlabel('f_min')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'f_min_hist.png'))

    pl.figure()
    pl.hist(f_maxs, bins=100, color='0.5')
    pl.ylabel('Count')
    pl.xlabel('f_max')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'f_max_hist.png'))

    pl.figure()
    pl.hist(taus, bins=100, color='0.5')
    pl.ylabel('Count')
    pl.xlabel('tau')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'tau_hist.png'))

    pl.figure()
    pl.hist(RMSE, bins=100, color='0.5')
    pl.ylabel('Count')
    pl.xlabel('RMSE')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'RMSE_hist.png'))

    data = pd.DataFrame(np.array([f_mins, f_maxs]).T, columns=['f_min', 'f_max'])
    jp = sns.jointplot('f_min', 'f_max', data=data, stat_func=None, color='0.5')
    jp.fig.set_size_inches(6.4, 4.8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'f_min_f_max_hist.png'))

    data = pd.DataFrame(np.array([f_mins, taus]).T, columns=['f_min', 'tau'])
    jp = sns.jointplot('f_min', 'tau', data=data, stat_func=None, color='0.5')
    jp.fig.set_size_inches(6.4, 4.8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'f_min_tau_hist.png'))

    data = pd.DataFrame(np.array([taus, f_maxs]).T, columns=['tau', 'f_max'])
    jp = sns.jointplot('tau', 'f_max', data=data, stat_func=None, color='0.5')
    jp.fig.set_size_inches(6.4, 4.8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'tau_f_max_hist.png'))
    pl.show()

    # TODO: cut off first ISI?