import os
import matplotlib.pyplot as pl
import numpy as np
import json
from cell_characteristics.fIcurve import compute_fIcurve, compute_fIcurve_last_ISI
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function, get_i_inj_standard_params
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.optimization.evaluation.plot_IV import plot_IV_traces, plot_fi_curve
from cell_characteristics.analyze_APs import get_AP_max_idx, get_AP_start_end
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    cell_id = '2015_08_26b'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # fI-curve for data
    protocol = 'IV'
    v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                     sweep_idxs=None, return_sweep_idxs=True)
    i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t_mat[0][-1], t_mat[0][1] - t_mat[0][0])
    params = get_i_inj_standard_params(protocol, sweep_idxs=sweep_idxs)
    amps = params['step_amp']
    start_step = params['start_step']
    end_step = params['end_step']
    # only take amps >= 0
    amps_greater0_idx = amps >= 0
    amps = amps[amps_greater0_idx]
    v_mat = v_mat[amps_greater0_idx]
    t_mat = t_mat[amps_greater0_idx]
    i_inj_mat = i_inj_mat[amps_greater0_idx]

    # fI-curve data
    firing_rates_data = compute_fIcurve(v_mat, t_mat[0], amps, start_step, end_step)
    firing_rates_data_last_ISI = compute_fIcurve_last_ISI(v_mat, t_mat[0], amps, start_step, end_step)

    # simulations model
    v_mat_model = list()
    simulation_params = get_standard_simulation_params()
    simulation_params['tstop'] = t_mat[0, -1]
    i_inj_mat_model = get_i_inj_from_function(protocol, np.array(sweep_idxs)[amps_greater0_idx], simulation_params['tstop'],
                                    simulation_params['dt'])
    for i in range(len(amps)):
        simulation_params['i_inj'] = i_inj_mat_model[i, :]
        v_model, t_model, _ = iclamp_handling_onset(cell, **simulation_params)
        v_mat_model.append(v_model)

    # fI-curve for model
    firing_rates_model = compute_fIcurve(v_mat_model, t_model, amps, start_step, end_step)
    firing_rates_model_last_ISI = compute_fIcurve_last_ISI(v_mat_model, t_model, amps, start_step, end_step)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'IV')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # save
    fi_dict = dict(amps=list(amps), firing_rates=list(firing_rates_model))
    with open(os.path.join(save_dir, 'img', 'IV', 'fi_curve', 'fi_dict.json'), 'w') as f:
        json.dump(fi_dict, f)

    np.save(os.path.join(save_dir_img, 'amps_greater0.npy'), amps)
    np.save(os.path.join(save_dir_img, 'firing_rates.npy'), firing_rates_model)

    #plot_fi_curve(amps, firing_rates_model, os.path.join(save_dir_img, 'fi_curve'))

    pl.figure()
    pl.plot(amps, firing_rates_data, '-ok', label='Exp. Data')
    pl.plot(amps, firing_rates_model, '-or', label='Model')
    pl.xlabel('Current (nA)')
    pl.ylabel('Frequency (Hz)')
    pl.legend(loc='upper right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'fIcurve.png'))

    pl.figure()
    pl.plot(amps, firing_rates_data_last_ISI, '-ok', label='Exp. Data')
    pl.plot(amps, firing_rates_model_last_ISI, '-or', label='Model')
    pl.xlabel('Current (nA)')
    pl.ylabel('last ISI (ms)')
    pl.legend(loc='upper right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'fIcurve_last_ISI.png'))
    pl.show()

    # # plot single traces
    # for amp, v_data, v_model in zip(amps, v_mat, v_mat_model):
    #     AP_start_model, AP_end_model = get_AP_start_end(v_model, threshold=-30, n=0)
    #     AP_start_data, AP_end_data = get_AP_start_end(v_data, threshold=-30, n=0)
    #
    #     pl.figure()
    #     if AP_start_model is not None and AP_start_data is not None:
    #         AP_peak_model = v_model[get_AP_max_idx(v_model, AP_start_model, AP_end_model)]
    #         AP_peak_data = v_data[get_AP_max_idx(v_data, AP_start_data, AP_end_data)]
    #         if AP_peak_model > AP_peak_data:
    #             pl.plot(t_model, v_model, 'r', label='Model')
    #             pl.plot(t_mat[0, :], v_data - v_data[0] + v_model[0], 'k', label='Exp. Data')
    #         else:
    #             pl.plot(t_mat[0, :], v_data - v_data[0] + v_model[0], 'k', label='Exp. Data')
    #             pl.plot(t_model, v_model, 'r', label='Model')
    #     else:
    #         pl.plot(t_mat[0, :], v_data - v_data[0] + v_model[0], 'k', label='Exp. Data')
    #         pl.plot(t_model, v_model, 'r', label='Model')
    #     pl.xlabel('Time (ms)')
    #     pl.ylabel('Membrane Potential (mV)')
    #     if amp == -0.1:
    #         pl.ylim(-80, -60)
    #     #pl.legend()
    #     pl.tight_layout()
    #     pl.savefig(os.path.join(save_dir_img, 'IV' + str(amp) + '.png'))
    #     pl.show()
    #
    # plot_IV_traces(amps[amps_greater0_idx], t_model, v_mat_model[amps_greater0_idx], save_dir_img)
    # pl.show()