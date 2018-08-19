import matplotlib.pyplot as pl
import numpy as np
import os
import json
from cell_characteristics.fIcurve import compute_fIcurve, compute_fIcurve_last_ISI
from cell_fitting.data.plot_IV import check_v_at_i_inj_0_is_at_right_sweep_idx
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj_from_function, \
    get_sweep_index_for_amp
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.util import init_nan
pl.style.use('paper')


def get_1st_ISI(v_traces, t_trace):
    dt = t_trace[1] - t_trace[0]

    ISI_1st = np.nan
    for v_trace in v_traces:
        AP_onsets = get_AP_onset_idxs(v_trace, threshold=0)
        if len(AP_onsets) > 3:
            ISI_1st = np.diff(AP_onsets)[0] * dt
            break
    return ISI_1st


def get_lag_1st_AP(v_traces, i_traces, t_trace):
    start_step = np.nonzero(i_traces[0])[0][0]
    dt = t_trace[1] - t_trace[0]

    lag_1st_AP = np.nan
    for v_trace in v_traces:
        AP_onsets = get_AP_onset_idxs(v_trace, threshold=0)
        if len(AP_onsets) >= 1:
            lag_1st_AP = (AP_onsets[0] - start_step) * dt
            break
    return lag_1st_AP


if __name__ == '__main__':

    # parameters
    save_dir = '../plots/IV/fi_curve/'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'IV'
    cells_ids = get_cells_for_protocol(data_dir, protocol)
    # cells_ids = ['2015_05_26d', '2015_06_08a', '2015_06_09f', '2015_06_19i', '2015_08_10g', '2015_08_26b']
    # cells_ids = ['2013_10_12a']
    animal = 'rat'

    ISI_1st = init_nan(len(cells_ids))
    lag_1st_AP = init_nan(len(cells_ids))
    for cell_idx, cell_id in enumerate(cells_ids):
        if not check_rat_or_gerbil(cell_id) == animal:
            continue

        # load data
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        t = t_mat[0, :]
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])

        try:
            check_v_at_i_inj_0_is_at_right_sweep_idx(v_mat, i_inj_mat)
        except AssertionError:
            continue

        # compute fi curve
        try:
            amps, firing_rates_data = compute_fIcurve(v_mat, i_inj_mat, t)
            _, firing_rates_data_last_ISI = compute_fIcurve_last_ISI(v_mat, i_inj_mat, t)
        except AssertionError:
            continue

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

        # get 1st ISI
        ISI_1st[cell_idx] = get_1st_ISI(v_mat, t)
        lag_1st_AP[cell_idx] = get_lag_1st_AP(v_mat, i_inj_mat, t)

        # plot
        save_dir_img = os.path.join(save_dir, animal, cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        # save
        if not os.path.exists(os.path.join(save_dir, cell_id)):
            os.makedirs(os.path.join(save_dir, cell_id))
        fi_dict = dict(amps=list(amps_greater0), firing_rates=list(firing_rates_data))
        with open(os.path.join(save_dir, cell_id, 'fi_dict.json'), 'w') as f:
            json.dump(fi_dict, f)

        pl.figure()
        pl.plot(amps_greater0, firing_rates_data, '-ok', label='Exp. Data')
        pl.xlabel('Current (nA)')
        pl.ylabel('Firing Rate (Hz)')
        #pl.legend(loc='lower right')
        pl.ylim(0, 100)
        pl.tight_layout()
        #pl.savefig(os.path.join(save_dir_img, 'fIcurve.png'))
        #pl.show()

        pl.figure()
        pl.plot(amps_greater0, firing_rates_data_last_ISI, '-ok', label='Exp. Data')
        pl.xlabel('Current (nA)')
        pl.ylabel('Last ISI (ms)')
        #pl.legend(loc='upper right')
        pl.tight_layout()
        #pl.savefig(os.path.join(save_dir_img, 'fIcurve_last_ISI.png'))
        #pl.show()

        # for amp, v_trace_data in zip(amps, v_mat):
        #     pl.figure()
        #     pl.plot(t, v_trace_data, 'k', label='Exp. Data')
        #     pl.xlabel('Time (ms)')
        #     pl.ylabel('Membrane Potential (mV)')
        #     #pl.legend(fontsize=16, loc='upper right')
        #     if np.round(amp, 2) == -0.1:
        #         pl.ylim(-80, -60)
        #     pl.tight_layout()
        #     pl.savefig(os.path.join(save_dir_img, 'IV_%.2f.png' % (amp)))
        #     #pl.show()
        #     pl.close()

        # # plot all traces in subplots
        # #fig, ax = pl.subplots(20, 1, sharex=True, figsize=(21, 29.7))
        # fig, ax = pl.subplots(20, 1, sharex=True, figsize=(8, 9))
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
        # #pl.savefig(os.path.join(save_dir_img, 'IV_subplots.pdf'))
        # pl.show()
        pl.close('all')

    # save
    save_dir_IV_characteristics = '../plots/IV/characteristics/'
    if not os.path.exists(save_dir_IV_characteristics):
        os.makedirs(save_dir_IV_characteristics)
    np.save(os.path.join(save_dir_IV_characteristics, 'cell_ids.npy'), cells_ids)
    np.save(os.path.join(save_dir_IV_characteristics, 'ISI_1st.npy'), ISI_1st)
    np.save(os.path.join(save_dir_IV_characteristics, 'lag_1st_AP.npy'), lag_1st_AP)