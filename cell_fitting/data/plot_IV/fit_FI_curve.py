from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
from cell_characteristics.analyze_step_current_data import compute_fIcurve
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj_from_function, \
    get_amp_for_sweep_index, get_i_inj_standard_params
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_fitting.data import check_cell_has_DAP
from cell_fitting.data.plot_IV import check_v_at_i_inj_0_is_at_right_sweep_idx
from cell_characteristics import to_idx
pl.style.use('paper')


def fit_fun(x, a, b, c):
    sr = a * (x - b)**c
    sr[x - b < 0] = 0
    return sr


if __name__ == '__main__':

    # parameters
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'IV'
    animal = 'rat'
    save_dir_img = os.path.join('../plots', protocol, 'fi_curve', animal)

    # get cell_ids
    cell_ids = get_cells_for_protocol(data_dir, protocol)
    cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)
    cell_ids = filter(lambda id: check_cell_has_DAP(id), cell_ids)

    FI_a = []
    FI_b = []
    FI_c = []
    cell_ids_used = []
    RMSE = []
    for cell_id in cell_ids:
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        t = t_mat[0, :]
        dt = t[1] -t [0]
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])
        params = get_i_inj_standard_params(protocol, sweep_idxs=sweep_idxs)
        amps = params['step_amp']
        start_step = params['start_step']
        end_step = params['end_step']

        try:
            check_v_at_i_inj_0_is_at_right_sweep_idx(v_mat, i_inj_mat, to_idx(start_step, dt), to_idx(end_step, dt))
        except AssertionError:
            continue
        if get_amp_for_sweep_index(sweep_idxs[-1], protocol) < 1.0:  # not enough positive amplitudes tested
            continue

        firing_rates_data = compute_fIcurve(v_mat, t, amps, start_step=start_step, end_step=end_step)

        # sort according to amplitudes
        idx_sort = np.argsort(amps)
        amps = amps[idx_sort]
        firing_rates_data = firing_rates_data[idx_sort]
        v_mat = v_mat[idx_sort]

        # only take amps >= 0
        amps_greater0_idx = amps >= 0
        amps_greater0 = amps[amps_greater0_idx]
        firing_rates_data = firing_rates_data[amps_greater0_idx]

        # fit square root to FI-curve
        # if firing_rates_data[-1] < 3/4 * np.max(firing_rates_data):
        #     pl.figure()
        #     pl.plot(amps_greater0, firing_rates_data)
        #
        #     pl.figure()
        #     pl.plot(t_mat[-3], v_mat[-3])
        #
        #     pl.figure()
        #     pl.plot(t_mat[20], v_mat[20])
        #     pl.show()
        #     continue
        firing_rates_data = firing_rates_data[:20]  # go up to 1 nA
        amps_greater0 = amps_greater0[:20]

        try:
            b0 = amps_greater0[np.where(firing_rates_data > 0)[0][0]]
            p_opt, _ = curve_fit(fit_fun, amps_greater0, firing_rates_data, p0=[50, b0, 0.5])
        except RuntimeError:
            continue
        if p_opt[0] <= 0:
            print 'a <= 0'
            continue
        rmse = np.sqrt(np.sum((firing_rates_data - fit_fun(amps_greater0, p_opt[0], p_opt[1], p_opt[2])) ** 2))
        if rmse > 20:
            continue
        # print 'a: ', p_opt[0]
        # print 'b: ', p_opt[1]
        # print 'c: ', p_opt[2]
        # pl.figure()
        # pl.plot(amps_greater0, firing_rates_data, color='k')
        # pl.plot(amps_greater0, fit_fun(amps_greater0, *p_opt), color='r')
        # pl.show()

        FI_a.append(p_opt[0])
        FI_b.append(p_opt[1])
        FI_c.append(p_opt[2])
        cell_ids_used.append(cell_id)
        RMSE.append(rmse)

        # print 'RMSE: %.5f' % RMSE[-1]
        # pl.figure()
        # pl.plot(amps_greater0, firing_rates_data, '-ok', label='Exp. Data')
        # pl.plot(amps_greater0, fit_fun(amps_greater0, p_opt[0], p_opt[1], p_opt[2]), 'b')
        # pl.xlabel('Current (nA)')
        # pl.ylabel('Firing Rate (Hz)')
        # pl.ylim(0, 100)
        # pl.tight_layout()
        # pl.show()

    # plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'FI_a.npy'), FI_a)
    np.save(os.path.join(save_dir_img, 'FI_b.npy'), FI_b)
    np.save(os.path.join(save_dir_img, 'FI_c.npy'), FI_c)
    np.save(os.path.join(save_dir_img, 'RMSE.npy'), RMSE)
    np.save(os.path.join(save_dir_img, 'cells.npy'), cell_ids_used)

    pl.figure()
    pl.hist(FI_a, bins=100, color='0.5')
    pl.ylabel('Count')
    pl.xlabel('Scaling')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'scaling_hist.png'))

    pl.figure()
    pl.hist(FI_b, bins=100, color='0.5')
    pl.ylabel('Count')
    pl.xlabel('Shift')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'shift_hist.png'))

    pl.figure()
    pl.hist(FI_c, bins=100, color='0.5')
    pl.ylabel('Count')
    pl.xlabel('Exponent')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'exponent_hist.png'))

    pl.figure()
    pl.hist(RMSE, bins=100, color='0.5')
    pl.ylabel('Count')
    pl.xlabel('RMSE')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'RMSE_hist.png'))

    data = pd.DataFrame(np.array([FI_a, FI_b]).T, columns=['Scaling', 'Shift'])
    jp = sns.jointplot('Scaling', 'Shift', data=data, stat_func=None, color='0.5')
    jp.fig.set_size_inches(6.4, 4.8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'scaling_shift_hist.png'))

    data = pd.DataFrame(np.array([FI_a, FI_c]).T, columns=['Scaling', 'Exponent'])
    jp = sns.jointplot('Scaling', 'Exponent', data=data, stat_func=None, color='0.5')
    jp.fig.set_size_inches(6.4, 4.8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'scaling_exponent_hist.png'))

    data = pd.DataFrame(np.array([FI_c, FI_b]).T, columns=['Exponent', 'Shift'])
    jp = sns.jointplot('Exponent', 'Shift', data=data, stat_func=None, color='0.5')
    jp.fig.set_size_inches(6.4, 4.8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'exponent_shift_hist.png'))

    data = pd.DataFrame(np.array([RMSE, FI_c]).T, columns=['RMSE', 'Exponent'])
    jp = sns.jointplot('RMSE', 'Exponent', data=data, stat_func=None, color='0.5')
    jp.fig.set_size_inches(6.4, 4.8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'exponent_shift_hist.png'))
    pl.show()