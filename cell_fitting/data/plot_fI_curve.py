import matplotlib.pyplot as pl
import numpy as np
import os
from scipy.optimize import curve_fit
from cell_characteristics.fIcurve import compute_fIcurve, compute_fIcurve_last_ISI
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    save_dir = './plots/fI_curve/rat'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'IV'
    v_rest_shift = -16
    cells = get_cells_for_protocol(data_dir, protocol)
    #cells = ['2015_05_26d', '2015_06_08a', '2015_06_09f', '2015_06_19i', '2015_08_10g', '2015_08_26b']
    animal = 'rat'

    for cell_id in cells:
        if not '2015' in cell_id:
            continue
        if not check_rat_or_gerbil(cell_id) == animal:
            continue

        # fI-curve for data
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        i_inj_mat = get_i_inj(protocol, sweep_idxs)
        t = t_mat[0, :]

        amps, firing_rates_data = compute_fIcurve(v_mat, i_inj_mat, t)
        _, firing_rates_data_last_ISI = compute_fIcurve_last_ISI(v_mat, i_inj_mat, t)

        # sort according to amplitudes
        idx_sort = np.argsort(amps)
        amps = amps[idx_sort]
        firing_rates_data = firing_rates_data[idx_sort]
        firing_rates_data_last_ISI = firing_rates_data_last_ISI[idx_sort]
        v_mat = v_mat[idx_sort]

        # only take amps >= 0
        amps_greater0_idx = amps >= 0
        amps_greater0 = amps[amps_greater0_idx]
        firing_rates_data = firing_rates_data[amps_greater0_idx]
        firing_rates_data_last_ISI = firing_rates_data_last_ISI[amps_greater0_idx]

        # plot
        save_dir_img = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        # pl.figure()
        # pl.plot(amps_greater0, firing_rates_data, '-ok', label='Exp. Data')
        # pl.xlabel('Current (nA)')
        # pl.ylabel('Firing rate (APs/ms)')
        # #pl.legend(loc='lower right')
        # pl.ylim(0, 0.09)
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_img, 'fIcurve.png'))
        # #pl.show()

        # fit square root to FI-curve
        def square_root(x, a, b):
            sr = np.sqrt(a * (x - b))
            sr[np.isnan(sr)] = 0
            return sr

        b0 = amps_greater0[np.where(firing_rates_data>0)[0][0]]
        p_opt, _ = curve_fit(square_root, amps_greater0, firing_rates_data, p0=[0.005, b0])
        print p_opt
        np.savetxt(os.path.join(save_dir_img, 'p_opt.txt'), p_opt)

        pl.figure()
        pl.plot(amps_greater0, firing_rates_data, '-ok', label='Exp. Data')
        pl.plot(amps_greater0, square_root(amps_greater0, p_opt[0], p_opt[1]), 'b')
        pl.xlabel('Current (nA)')
        pl.ylabel('Firing rate (APs/ms)')
        # pl.legend(loc='lower right')
        pl.ylim(0, 0.09)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'fIcurve_fit.png'))
        pl.show()

        # pl.figure()
        # pl.plot(amps_greater0, firing_rates_data_last_ISI, '-ok', label='Exp. Data')
        # pl.xlabel('Current (nA)')
        # pl.ylabel('Last ISI (ms)')
        # #pl.legend(loc='upper right')
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_img, 'fIcurve_last_ISI.png'))
        # #pl.show()
        #
        # for amp, v_trace_data in zip(amps, v_mat):
        #     pl.figure()
        #     pl.plot(t, v_trace_data, 'k', label='Exp. Data')
        #     pl.xlabel('Time (ms)')
        #     pl.ylabel('Membrane Potential (mV)')
        #     #pl.legend(fontsize=16, loc='upper right')
        #     if np.round(amp, 2) == -0.1:
        #         pl.ylim(-80, -60)
        #     pl.tight_layout()
        #     pl.savefig(os.path.join(save_dir_img, 'IV_' + str(amp) + '.png'))
        #     pl.show()
        #
        # # plot all traces in subplots
        # fig, ax = pl.subplots(20, 1, sharex=True, figsize=(21, 29.7))
        # for i, (amp, v_trace_data) in enumerate(
        #         zip(amps[amps_greater0_idx][1:21], v_mat[amps_greater0_idx][1:21])):
        #     ax[i].plot(t, v_trace_data, 'r', label='$i_{amp}: $ %.2f' % amp)
        #     ax[i].set_ylim(-80, 60)
        #     ax[i].set_xlim(200, 850)
        #     ax[i].legend(fontsize=14)
        # # pl.tight_layout()
        # fig.text(0.06, 0.5, 'Membrane Potential (mV)', va='center', rotation='vertical', fontsize=14)
        # fig.text(0.5, 0.06, 'Time (ms)', ha='center', fontsize=14)
        # #pl.savefig(os.path.join(save_dir_img, 'IV_subplots.png'))
        # pl.savefig(os.path.join(save_dir_img, 'IV_subplots.pdf'))
        # pl.show()