from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from scipy.optimize import curve_fit
from cell_characteristics.fIcurve import compute_fIcurve
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj_from_function
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
import seaborn as sns
import pandas as pd
pl.style.use('paper')


def square_root(x, a, b):
    sr = np.sqrt(a * (x - b))
    sr[np.isnan(sr)] = 0
    return sr


if __name__ == '__main__':

    # parameters
    save_dir = './plots/fI_curve/rat'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'IV'
    cells = get_cells_for_protocol(data_dir, protocol)
    animal = 'rat'
    FI_a = []
    FI_b = []
    Cells = []

    for cell_id in cells:
        # if not '2015' in cell_id:
        #     continue
        if not check_rat_or_gerbil(cell_id) == animal:
            continue

        # fI-curve for data
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        t = t_mat[0, :]
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])

        try:
            amps, firing_rates_data = compute_fIcurve(v_mat, i_inj_mat, t)
        except IndexError:
            continue

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
        if len(firing_rates_data) < 21 or firing_rates_data[-1] < 3/4 * np.max(firing_rates_data):
            continue
        firing_rates_data = firing_rates_data[:21]  # go to 1 nA
        amps_greater0 = amps_greater0[:21]
        try:
            b0 = amps_greater0[np.where(firing_rates_data > 0)[0][0]]
        except IndexError:
            continue
        try:
            p_opt, _ = curve_fit(square_root, amps_greater0, firing_rates_data, p0=[0.005, b0])
        except RuntimeError:
            continue
        if p_opt[0] <= 0:
            continue

        FI_a.append(p_opt[0])
        FI_b.append(p_opt[1])
        Cells.append(cell_id)

        # pl.figure()
        # pl.plot(amps_greater0, firing_rates_data, '-ok', label='Exp. Data')
        # pl.plot(amps_greater0, square_root(amps_greater0, p_opt[0], p_opt[1]), 'b')
        # pl.xlabel('Current (nA)')
        # pl.ylabel('Firing rate (APs/ms)')
        # # pl.legend(loc='lower right')
        # pl.ylim(0, 0.09)
        # pl.tight_layout()
        # pl.show()

    # plot
    save_dir_img = os.path.join(save_dir, 'summary')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'FI_a.npy'), FI_a)
    np.save(os.path.join(save_dir_img, 'FI_b.npy'), FI_b)
    np.save(os.path.join(save_dir_img, 'cells.npy'), Cells)

    pl.figure()
    pl.hist(FI_a, bins=100, color='0.5')
    pl.ylabel('Count')
    pl.xlabel('Scaling')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'scaling_hist.png'))
    #pl.show()

    pl.figure()
    pl.hist(FI_b, bins=100, color='0.5')
    pl.ylabel('Count')
    pl.xlabel('Shift')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'shift_hist.png'))
    #pl.show()

    data = pd.DataFrame(np.array([FI_a, FI_b]).T, columns=['Scaling', 'Shift'])
    jp = sns.jointplot('Scaling', 'Shift', data=data, stat_func=None, color='0.5') #, xlim=(0, 0.025), ylim=(0, 0.8))
    jp.fig.set_size_inches(6.4, 4.8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'scaling_shift_hist.png'))
    pl.show()

    # TODO: sort out interneurons