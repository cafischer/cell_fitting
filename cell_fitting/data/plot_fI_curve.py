import matplotlib.pyplot as pl
import numpy as np
import os
from cell_characteristics.fIcurve import compute_fIcurve, compute_fIcurve_last_ISI
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj


if __name__ == '__main__':

    # parameters
    save_dir = './plots/fI_curve'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'IV'
    v_rest_shift = -16
    cells = get_cells_for_protocol(data_dir, protocol)

    for cell  in cells:
        # fI-curve for data
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell), protocol, return_sweep_idxs=True)
        i_inj_mat = get_i_inj(protocol, sweep_idxs)
        t = t_mat[0, :]

        amps, firing_rates_data = compute_fIcurve(v_mat, i_inj_mat, t)
        _, firing_rates_data_last_ISI  = compute_fIcurve_last_ISI(v_mat, i_inj_mat, t)

        # sort according to amplitudes
        idx_sort = np.argsort(amps)
        amps = amps[idx_sort]
        firing_rates_data = firing_rates_data[idx_sort]
        firing_rates_data_last_ISI = firing_rates_data_last_ISI[idx_sort]
        v_traces_data = np.array(v_traces_data)[idx_sort]

        # only take amps >= 0
        amps_greater0_idx = amps >= 0
        amps_greater0 = amps[amps_greater0_idx]
        firing_rates_data = firing_rates_data[amps_greater0_idx]
        firing_rates_data_last_ISI = firing_rates_data_last_ISI[amps_greater0_idx]

        # plot
        save_dir_fig = os.path.join(save_dir, 'img/IV')
        if not os.path.exists(save_dir_fig):
            os.makedirs(save_dir_fig)

        pl.figure()
        pl.plot(amps_greater0, firing_rates_data, 'k', label='Exp. Data')
        pl.xlabel('Current (nA)', fontsize=16)
        pl.ylabel('Firing rate (APs/ms)', fontsize=16)
        pl.legend(loc='lower right', fontsize=16)
        pl.savefig(os.path.join(save_dir_fig, 'fIcurve.png'))
        pl.show()

        pl.figure()
        pl.plot(amps_greater0, firing_rates_data_last_ISI, 'k', label='Exp. Data')
        pl.xlabel('Current (nA)', fontsize=16)
        pl.ylabel('last ISI (ms)', fontsize=16)
        pl.legend(loc='upper right', fontsize=16)
        pl.savefig(os.path.join(save_dir_fig, 'fIcurve_last_ISI.png'))
        pl.show()

        for amp, v_trace_data in zip(amps, v_traces_data):
            pl.figure()
            pl.plot(t, v_trace_data, 'k', label='Exp. Data')
            pl.xlabel('Time (ms)', fontsize=16)
            pl.ylabel('Membrane Potential (mV)', fontsize=16)
            pl.legend(fontsize=16, loc='upper right')
            pl.savefig(os.path.join(save_dir_fig, 'IV'+str(amp)+'.png'))
            #pl.show()