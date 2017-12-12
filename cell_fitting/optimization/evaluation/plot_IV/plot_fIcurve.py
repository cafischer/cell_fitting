import os
import matplotlib.pyplot as pl
import numpy as np
from cell_characteristics.fIcurve import compute_fIcurve, compute_fIcurve_last_ISI
from nrn_wrapper import Cell
from cell_fitting.optimization.fitter import extract_simulation_params
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.util import merge_dicts
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    #save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    save_dir = '/home/cf/Phd/server/cns/server/results/sensitivity_analysis/2017-10-10_14:00:42/76805'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    cell_id = '2015_08_26b'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # fI-curve for data
    protocol = 'plot_IV'
    v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                     sweep_idxs=None, return_sweep_idxs=True)
    i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t_mat[0][-1], t_mat[0][1] - t_mat[0][0])
    amps, firing_rates_data = compute_fIcurve(v_mat, i_inj_mat, t_mat[0])
    _, firing_rates_data_last_ISI = compute_fIcurve_last_ISI(v_mat, i_inj_mat, t_mat[0])

    # fI-curve for model
    v_mat_model = list()
    for i in range(len(sweep_idxs)):
        sim_params = {'celsius': 35, 'onset': 200}
        simulation_params = merge_dicts(extract_simulation_params(v_mat[i], t_mat[i], i_inj_mat[i]), sim_params)
        v_model, t_model, _ = iclamp_handling_onset(cell, **simulation_params)
        v_mat_model.append(v_model)

    amps, firing_rates_model = compute_fIcurve(v_mat_model, i_inj_mat, t_mat[0])
    _, firing_rates_model_last_ISI = compute_fIcurve_last_ISI(v_mat_model, i_inj_mat, t_mat[0])

    # sort according to amplitudes
    idx_sort = np.argsort(amps)
    amps = amps[idx_sort]
    firing_rates_data = firing_rates_data[idx_sort]
    firing_rates_model = firing_rates_model[idx_sort]
    firing_rates_data_last_ISI = firing_rates_data_last_ISI[idx_sort]
    firing_rates_model_last_ISI = firing_rates_model_last_ISI[idx_sort]
    v_traces_data = np.array(v_mat)[idx_sort]
    v_mat_model = np.array(v_mat_model)[idx_sort]

    # only take amps >= 0
    amps_greater0_idx = amps >= 0
    amps_greater0 = amps[amps_greater0_idx]
    firing_rates_data = firing_rates_data[amps_greater0_idx]
    firing_rates_model = firing_rates_model[amps_greater0_idx]
    firing_rates_data_last_ISI = firing_rates_data_last_ISI[amps_greater0_idx]
    firing_rates_model_last_ISI = firing_rates_model_last_ISI[amps_greater0_idx]
    #v_traces_data = v_traces_data[amps_greater0]
    #v_traces_model = v_traces_model[amps_greater0]

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'plot_IV')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    pl.figure()
    #pl.plot(amps_greater0, firing_rates_data, '-ok', label='Exp. Data')
    pl.plot(amps_greater0, firing_rates_model, '-or', label='Model')
    pl.ylim([0, 0.09])
    pl.xlabel('Current (nA)')
    pl.ylabel('Firing rate (APs/ms)')
    #pl.legend(loc='lower right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'fIcurve.png'))
    #pl.show()

    np.save(os.path.join(save_dir_img, 'amps_greater0.npy'), amps_greater0)
    np.save(os.path.join(save_dir_img, 'firing_rates.npy'), firing_rates_model)

    pl.figure()
    pl.plot(amps_greater0, firing_rates_data_last_ISI, '-ok', label='Exp. Data')
    pl.plot(amps_greater0, firing_rates_model_last_ISI, '-or', label='Model')
    pl.xlabel('Current (nA)')
    pl.ylabel('last ISI (ms)')
    pl.legend(loc='upper right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'fIcurve_last_ISI.png'))
    #pl.show()

    # # plot single traces
    # for amp, v_trace_data, v_trace_model in zip(amps, v_traces_data, v_traces_model):
    #     AP_start_model, AP_end_model = get_AP_start_end(v_trace_model, threshold=-30, n=0)
    #     AP_start_data, AP_end_data = get_AP_start_end(v_trace_data, threshold=-30, n=0)
    #
    #     pl.figure()
    #     if AP_start_model is not None and AP_start_data is not None:
    #         AP_peak_model = v_trace_model[get_AP_max_idx(v_trace_model, AP_start_model, AP_end_model)]
    #         AP_peak_data = v_trace_data[get_AP_max_idx(v_trace_data, AP_start_data, AP_end_data)]
    #         if AP_peak_model > AP_peak_data:
    #             pl.plot(t_model, v_trace_model, 'r', label='Model')
    #             #pl.plot(t_trace, v_trace_data, 'k', label='Exp. Data')
    #         else:
    #             #pl.plot(t_trace, v_trace_data, 'k', label='Exp. Data')
    #             pl.plot(t_model, v_trace_model, 'r', label='Model')
    #     else:
    #         #pl.plot(t_trace, v_trace_data, 'k', label='Exp. Data')
    #         pl.plot(t_model, v_trace_model, 'r', label='Model')
    #     pl.xlabel('Time (ms)')
    #     pl.ylabel('Membrane Potential (mV)')
    #     if amp == -0.1:
    #         pl.ylim(-80, -60)
    #     #pl.legend()
    #     pl.tight_layout()
    #     pl.savefig(os.path.join(save_dir_img, 'plot_IV' + str(amp) + '.png'))
    #     pl.show()

    # plot all under another with subfigures
    #fig, ax = pl.subplots(sum(amps_greater0_idx), 1, sharex=True)
    fig, ax = pl.subplots(20, 1, sharex=True, figsize=(21, 29.7))
    for i, (amp, v_trace_model) in enumerate(zip(amps[amps_greater0_idx][1:21], v_mat_model[amps_greater0_idx][1:21])):
        ax[i].plot(t_model, v_trace_model, 'r', label='$i_{amp}: $ %.2f' % amp)
        ax[i].set_ylim(-80, 60)
        ax[i].set_xlim(200, 850)
        ax[i].legend(fontsize=14)
    #pl.tight_layout()
    fig.text(0.06, 0.5, 'Membrane Potential (mV)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.06, 'Time (ms)', ha='center', fontsize=14)
    pl.savefig(os.path.join(save_dir_img, 'IV_subplots.pdf'))
    pl.show()