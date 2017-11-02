import matplotlib.pyplot as pl
import numpy as np
import os
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
    #cells = get_cells_for_protocol(data_dir, protocol)[20:]
    cells = ['2015_05_26d', '2015_06_08a', '2015_06_09f', '2015_06_19i', '2015_08_10g', '2015_08_26b']
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
        save_dir_fig = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_fig):
            os.makedirs(save_dir_fig)

        pl.figure()
        pl.plot(amps_greater0, firing_rates_data, '-ok', label='Exp. Data')
        pl.xlabel('Current (nA)')
        pl.ylabel('Firing rate (APs/ms)')
        #pl.legend(loc='lower right')
        pl.ylim(0, 0.09)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_fig, 'fIcurve.png'))
        #pl.show()

        pl.figure()
        pl.plot(amps_greater0, firing_rates_data_last_ISI, '-ok', label='Exp. Data')
        pl.xlabel('Current (nA)')
        pl.ylabel('Last ISI (ms)')
        #pl.legend(loc='upper right')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_fig, 'fIcurve_last_ISI.png'))
        #pl.show()

        for amp, v_trace_data in zip(amps, v_mat):
            pl.figure()
            pl.plot(t, v_trace_data, 'k', label='Exp. Data')
            pl.xlabel('Time (ms)')
            pl.ylabel('Membrane Potential (mV)')
            #pl.legend(fontsize=16, loc='upper right')
            if np.round(amp, 2) == -0.1:
                pl.ylim(-80, -60)
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_fig, 'IV_'+str(amp)+'.png'))
            pl.show()