import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.optimization.simulate import iclamp_adaptive_handling_onset
from nrn_wrapper import Cell
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj
pl.style.use('paper')


def compute_v_sag_and_steady_state(v_traces, amps, AP_threshold, start_step, end_step):

    amps_subtheshold = []
    v_steady_states = []
    v_sags = []

    for i, v_trace in enumerate(v_traces):
        if len(get_AP_onset_idxs(v_trace, AP_threshold)) == 0:
            amps_subtheshold.append(amps[i])
            v_steady_state = np.mean(v_trace[end_step - int(np.round((end_step - start_step) / 4)):end_step])
            v_steady_states.append(v_steady_state)
            if amps[i] > 0:
                v_sag = np.max(v_trace[start_step:start_step + int(np.round((end_step - start_step) / 4))])
            else:
                v_sag = np.min(v_trace[start_step:start_step + int(np.round((end_step - start_step) / 4))])
            v_sags.append(v_sag)

            # print 'v_steady_state: %.2f' % v_steady_state
            # print 'v_sag: %.2f' % v_sag
            # pl.figure()
            # pl.plot(v_trace)
            # pl.show()
    return v_sags, v_steady_states, amps_subtheshold


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
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

    # discontinuities for IV
    dt = 0.05
    start_step = int(round(250 / dt))
    end_step = int(round(750 / dt))
    discontinuities_IV = [start_step, end_step]

    # read data
    v_mat_data, t_mat_data, sweep_idxs = get_v_and_t_from_heka(data_dir, 'IV', return_sweep_idxs=True)
    i_inj_mat = get_i_inj('IV', sweep_idxs)

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

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'IV', 'sag_vs_steady_state')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    pl.figure()
    pl.plot(amps_subtheshold, v_steady_states, linestyle='-', marker='o', c='0.0', label='Steady State')
    pl.plot(amps_subtheshold, v_sags, linestyle='-', marker='o', c='0.5', label='Sag')
    pl.xlabel('Current (nA)')
    pl.ylabel('Membrane Potential (mV)')
    pl.legend(loc='upper left')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'sag_steady_state.png'))
    pl.show()