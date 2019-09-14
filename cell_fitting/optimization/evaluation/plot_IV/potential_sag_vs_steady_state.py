import os
import matplotlib.pyplot as pl
import numpy as np
import json
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import iclamp_adaptive_handling_onset, get_standard_simulation_params
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function, get_i_inj_standard_params
from cell_characteristics import to_idx
pl.style.use('paper')


def compute_v_sag_and_steady_state(v_traces, amps, AP_threshold, start_step_idx, end_step_idx):

    amps_subtheshold = []
    v_steady_states = []
    v_sags = []

    for i, v_trace in enumerate(v_traces):
        onset_idxs = get_AP_onset_idxs(v_trace, AP_threshold)
        onset_idxs = onset_idxs[onset_idxs <= end_step_idx]  # rebound spikes are allowed
        if len(onset_idxs) == 0:
            amps_subtheshold.append(amps[i])
            v_steady_state = np.mean(v_trace[end_step_idx - int(np.round((end_step_idx - start_step_idx) / 4)):end_step_idx])
            v_steady_states.append(v_steady_state)
            if amps[i] > 0:
                v_sag = np.max(v_trace[start_step_idx:start_step_idx + int(np.round((end_step_idx - start_step_idx) / 4))])
            else:
                v_sag = np.min(v_trace[start_step_idx:start_step_idx + int(np.round((end_step_idx - start_step_idx) / 4))])
            v_sags.append(v_sag)

            # print 'v_steady_state: %.2f' % v_steady_state
            # print 'v_sag: %.2f' % v_sag
            # pl.figure()
            # pl.plot(v_trace)
            # pl.show()
    return v_sags, v_steady_states, amps_subtheshold


def plot_sag_vs_steady_state_on_ax(ax, amps_subtheshold, v_steady_states, v_sags, color_lines='k', label=True):
    ax.plot(amps_subtheshold, v_steady_states, linestyle='-', marker='s', c=color_lines, markersize=4,
            label='Steady State' if label else '')
    ax.plot(amps_subtheshold, v_sags, linestyle='-', marker='$\cup$', c=color_lines, alpha=0.5, markersize=4,
            label='Sag' if label else '')
    ax.set_xlabel('Current (nA)')
    #ax.set_ylabel('Mem. pot. (mV)')
    #ax.legend(loc='upper left')
    ax.set_xticks(np.arange(-0.15, 0.15+0.1, 0.1))


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cfischer/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell_rounded.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    data_dir = '/home/cfischer/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'
    AP_threshold = -20
    v_shift = -16
    protocol = 'IV'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # read data
    v_mat_data, t_mat_data, sweep_idxs = get_v_and_t_from_heka(data_dir, 'IV', return_sweep_idxs=True)
    params_IV = get_i_inj_standard_params(protocol, sweep_idxs)
    amps = params_IV['step_amp']
    start_step = params_IV['start_step']
    end_step = params_IV['end_step']

    # IV for model
    simulation_params = get_standard_simulation_params()
    simulation_params['tstop'] = np.round(t_mat_data[0, -1])
    i_inj_mat = get_i_inj_from_function('IV', sweep_idxs, simulation_params['tstop'], simulation_params['dt'])
    v_traces_model = []
    for i in range(len(i_inj_mat)):
        simulation_params['i_inj'] = i_inj_mat[i]
        v_model, t_model, _ = iclamp_adaptive_handling_onset(cell, **simulation_params)
        v_traces_model.append(v_model)

    v_sags, v_steady_states, amps_subtheshold = compute_v_sag_and_steady_state(v_traces_model, amps, AP_threshold,
                                                                               to_idx(start_step, simulation_params['dt']),
                                                                               to_idx(end_step, simulation_params['dt']))

    # save
    max_amp = 0.15
    amps_subtheshold_bool = np.array(amps_subtheshold) < max_amp + 0.05
    amps_subtheshold = np.array(amps_subtheshold)[amps_subtheshold_bool]
    v_steady_states = np.array(v_steady_states)[amps_subtheshold_bool]
    v_sags = np.array(v_sags)[amps_subtheshold_bool]

    sag_dict = dict(amps_subtheshold=list(amps_subtheshold), v_steady_states=list(v_steady_states), v_sags=list(v_sags))
    if not os.path.exists(os.path.join(save_dir, 'img', 'IV', 'sag')):
        os.makedirs(os.path.join(save_dir, 'img', 'IV', 'sag'))
    with open(os.path.join(save_dir, 'img', 'IV', 'sag', 'sag_dict.json'), 'w') as f:
        json.dump(sag_dict, f)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'IV', 'sag')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    fig, ax = pl.subplots()
    plot_sag_vs_steady_state_on_ax(ax, amps_subtheshold, v_steady_states, v_sags)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'sag_vs_steady_state.png'))
    pl.show()