import matplotlib.pyplot as pl
import numpy as np
import os
from cell_characteristics.fIcurve import compute_fIcurve, compute_fIcurve_last_ISI
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj_from_function, get_sweep_index_for_amp
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
pl.style.use('paper')


def get_index_i_inj_start_end(i_inj):
    nonzero = np.nonzero(i_inj)[0]
    if len(nonzero) <= 1:
        return None
    else:
        return nonzero[0], nonzero[-1]


if __name__ == '__main__':

    # parameters
    save_dir = '../plots/plot_IV/fi_curve/rat'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'plot_IV'
    #cells = get_cells_for_protocol(data_dir, protocol)
    cells = ['2013_12_02a']
    #cells = ['2015_05_26d', '2015_06_08a', '2015_06_09f', '2015_06_19i', '2015_08_10g', '2015_08_26b']
    animal = 'rat'

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

        # check that no step in V at 0 injected current
        i_inj_start_idx, i_inj_end_idx = get_index_i_inj_start_end(i_inj_mat[0, :])
        sweep0 = get_sweep_index_for_amp(0, 'plot_IV')
        try:
            if np.abs(np.mean(np.concatenate((v_mat[sweep0, :i_inj_start_idx], v_mat[sweep0, i_inj_end_idx:])))
                                                     - np.mean(v_mat[sweep0, i_inj_start_idx:i_inj_end_idx])) >= 0.5:  # mV mean values should be roughly the same
                print '0 trace not ok: ' + str(cell_id)
        except IndexError:
            continue

        # compute fi curve
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

        pl.figure()
        pl.plot(amps_greater0, firing_rates_data, '-ok', label='Exp. Data')
        pl.xlabel('Current (nA)')
        pl.ylabel('Firing Rate (Hz)')
        #pl.legend(loc='lower right')
        pl.ylim(0, 100)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'fIcurve.png'))
        #pl.show()

        pl.figure()
        pl.plot(amps_greater0, firing_rates_data_last_ISI, '-ok', label='Exp. Data')
        pl.xlabel('Current (nA)')
        pl.ylabel('Last ISI (ms)')
        #pl.legend(loc='upper right')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'fIcurve_last_ISI.png'))
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
            pl.savefig(os.path.join(save_dir_img, 'IV_%.2f.png' % (amp)))
            #pl.show()
            pl.close()

        # plot all traces in subplots
        fig, ax = pl.subplots(20, 1, sharex=True, figsize=(21, 29.7))
        for i, (amp, v_trace_data) in enumerate(
                zip(amps[amps_greater0_idx][1:21], v_mat[amps_greater0_idx][1:21])):
            ax[i].plot(t, v_trace_data, 'r', label='$i_{amp}: $ %.2f' % amp)
            ax[i].set_ylim(-80, 60)
            ax[i].set_xlim(200, 850)
            ax[i].legend(fontsize=14)
        # pl.tight_layout()
        fig.text(0.06, 0.5, 'Membrane Potential (mV)', va='center', rotation='vertical', fontsize=14)
        fig.text(0.5, 0.06, 'Time (ms)', ha='center', fontsize=14)
        #pl.savefig(os.path.join(save_dir_img, 'IV_subplots.png'))
        pl.savefig(os.path.join(save_dir_img, 'IV_subplots.pdf'))
        #pl.show()