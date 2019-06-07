import matplotlib.pyplot as pl
import numpy as np
import os
import json
from cell_characteristics.analyze_step_current_data import compute_fIcurve, compute_fIcurve_last_ISI
from cell_fitting.data.plot_IV import check_v_at_i_inj_0_is_at_right_sweep_idx
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj_from_function, \
    get_sweep_index_for_amp, get_i_inj_standard_params
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_characteristics import to_idx
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    save_dir = '../plots/IV/fi_curve/'
    data_dir = '/media/cfischer/TOSHIBA EXT/Sicherung_2018_05_19/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'IV'
    animal = 'rat'

    # get cell_ids
    cell_ids = get_cells_for_protocol(data_dir, protocol)
    #cell_ids = ['2015_05_26d', '2015_06_08a', '2015_06_09f', '2015_06_19i', '2015_08_10g', '2015_08_26b']
    cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)


    for cell_idx, cell_id in enumerate(cell_ids):
        if not check_rat_or_gerbil(cell_id) == animal:
            continue

        # load data
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        t = t_mat[0, :]
        dt = t[1] - t[0]
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])
        params = get_i_inj_standard_params(protocol, sweep_idxs=sweep_idxs)
        amps = params['step_amp']
        start_step = params['start_step']
        end_step = params['end_step']
        start_step_idx = to_idx(start_step, dt)
        try:
            check_v_at_i_inj_0_is_at_right_sweep_idx(v_mat, i_inj_mat, start_step_idx, to_idx(end_step, dt))
        except AssertionError:
            continue

        # only take amps >= 0
        amps_greater0_idx = amps >= 0
        amps = amps[amps_greater0_idx]
        v_mat = v_mat[amps_greater0_idx]
        t_mat = t_mat[amps_greater0_idx]
        i_inj_mat = i_inj_mat[amps_greater0_idx]

        # compute FI curve
        try:
            firing_rates_data = compute_fIcurve(v_mat, t, amps, start_step, end_step)
            firing_rates_data_last_ISI = compute_fIcurve_last_ISI(v_mat, t, amps, start_step, end_step)
        except AssertionError:
            continue

        # plot
        save_dir_img = os.path.join(save_dir, animal, cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        # save
        fi_dict = dict(amps=list(amps), firing_rates=list(firing_rates_data))
        with open(os.path.join(save_dir_img, 'fi_dict.json'), 'w') as f:
            json.dump(fi_dict, f)

        # pl.figure()
        # pl.plot(amps, firing_rates_data, '-ok', label='Exp. Data')
        # pl.xlabel('Current (nA)')
        # pl.ylabel('Firing Rate (Hz)')
        # #pl.legend(loc='lower right')
        # pl.ylim(0, 80)
        # pl.tight_layout()
        # #pl.savefig(os.path.join(save_dir_img, 'fIcurve.png'))
        # pl.show()

        # pl.figure()
        # pl.plot(amps, firing_rates_data_last_ISI, '-ok', label='Exp. Data')
        # pl.xlabel('Current (nA)')
        # pl.ylabel('Last ISI (ms)')
        # #pl.legend(loc='upper right')
        # pl.tight_layout()
        # #pl.savefig(os.path.join(save_dir_img, 'fIcurve_last_ISI.png'))
        # #pl.show()

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
        pl.show()
        pl.close('all')