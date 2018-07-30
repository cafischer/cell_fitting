import os
import matplotlib.pyplot as pl
import numpy as np
import json
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import iclamp_adaptive_handling_onset
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function
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


def plot_sag_vs_steady_state_on_ax(ax, amps_subtheshold, v_steady_states, v_sags):
    ax.plot(amps_subtheshold, v_steady_states, linestyle='-', marker='o', c='0.0', markersize=4, label='Steady State')
    ax.plot(amps_subtheshold, v_sags, linestyle='-', marker='o', c='0.5', markersize=4, label='Sag')
    ax.set_xlabel('Inj. current (nA)', fontsize=12)
    ax.set_ylabel('Mem. pot. (mV)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xticks(np.arange(-0.15, 0.15+0.05, 0.05))
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    #save_dir = '../../../results/server/2017-07-27_09:18:59/22/L-BFGS-B'
    #model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../../results/hand_tuning/test0'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'

    AP_threshold = -30
    v_shift = -16

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # discontinuities for plot_IV
    dt = 0.05
    start_step = int(round(250 / dt))
    end_step = int(round(750 / dt))
    discontinuities_IV = [start_step, end_step]

    # read data
    v_mat_data, t_mat_data, sweep_idxs = get_v_and_t_from_heka(data_dir, 'IV', return_sweep_idxs=True)
    i_inj_mat = get_i_inj_from_function('IV', sweep_idxs, t_mat_data[0][-1], t_mat_data[0][1]-t_mat_data[0][0])

    # VI for model
    simulation_params = {'sec': ('soma', None), 'celsius': 35, 'onset': 200, 'atol': 1e-6, 'continuous': True,
                         'discontinuities': discontinuities_IV, 'interpolate': True,
                         'v_init': v_mat_data[0, 0] + v_shift, 'tstop': t_mat_data[0, -1],
                         'dt': t_mat_data[0, 1] - t_mat_data[0, 0]}
    #simulation_params = {'celsius': 35, 'onset': 200}

    v_traces_model = []
    for i in range(len(v_mat_data)):
        simulation_params['i_inj'] = i_inj_mat[i]
        v_model, t_model, _ = iclamp_adaptive_handling_onset(cell, **simulation_params)
        v_traces_model.append(v_model)

    # compute amplitudes
    start_step = np.nonzero(i_inj_mat[0])[0][0]
    end_step = np.nonzero(i_inj_mat[0])[0][-1] + 1
    amps = np.array([i_inj[start_step] for i_inj in i_inj_mat])

    # sort according to amplitudes
    idx_sort = np.argsort(amps)
    amps = amps[idx_sort]
    v_traces_model = np.array(v_traces_model)[idx_sort]

    v_sags, v_steady_states, amps_subtheshold = compute_v_sag_and_steady_state(v_traces_model, amps, AP_threshold,
                                                                               start_step, end_step)

    # save
    max_amp = 0.15
    amps_subtheshold_range = np.array(amps_subtheshold) < max_amp + 0.05
    amps_subtheshold = np.array(amps_subtheshold)[amps_subtheshold_range]
    v_steady_states = np.array(v_steady_states)[amps_subtheshold_range]
    v_sags = np.array(v_sags)[amps_subtheshold_range]

    sag_dict = dict(amps_subtheshold=list(amps_subtheshold), v_steady_states=list(v_steady_states), v_sags=list(v_sags))
    with open(os.path.join(save_dir, 'img', 'IV', 'sag', 'sag_dict.json'), 'w') as f:
        json.dump(sag_dict, f)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'IV', 'sag')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    fig, ax = pl.subplots()
    plot_sag_vs_steady_state_on_ax(ax, amps_subtheshold_range, v_steady_states, v_sags)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'sag_vs_steady_state.png'))