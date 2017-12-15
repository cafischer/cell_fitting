from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
import pandas as pd
from cell_fitting.data import shift_v_rest
from cell_fitting.read_heka import get_v_and_t_from_heka, get_protocols_same_base
from cell_characteristics.analyze_APs import get_spike_characteristics, get_AP_onset_idxs
from cell_characteristics import to_idx
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_fitting.util import init_nan
pl.style.use('paper')


def get_array_from_str(array_str):
    array_str = array_str.replace('[', '').replace(']', '')
    array = np.array([float(x) for x in array_str.split(', ')])
    return array


def find_series_idx(series_nr_to_find, series_traces):
    idxs = np.zeros(len(series_nr_to_find), dtype=int)
    for i, series_nr in enumerate(series_nr_to_find):
        idxs[i] = np.where('Series' + str(int(series_nr)) == np.array(series_traces))[0]
    return idxs


def get_amps_v_traces_t_traces(cell_id, data_dir, protocol_hyper, protocol_depo, hyper_depo_params):
    v_traces = []
    t_traces = []
    series_traces = []
    cell_dir = os.path.join(data_dir, cell_id + '.dat')
    protocols_match, heka_dict = get_protocols_same_base(cell_dir, protocol_hyper, return_heka=True)
    for protocol in protocols_match:
        v_mat, t_mat, series = get_v_and_t_from_heka(cell_dir, protocol, sweep_idxs=[0], return_series=True,
                                                     heka_dict=heka_dict)
        v = shift_v_rest(v_mat[0], v_rest_shift)
        t = t_mat[0]
        v_traces.append(v)
        t_traces.append(t)
        series_traces.append(series)
    protocols_match, heka_dict = get_protocols_same_base(cell_dir, protocol_depo, return_heka=True)
    for protocol in protocols_match:
        v_mat, t_mat, series = get_v_and_t_from_heka(cell_dir, protocol, sweep_idxs=[0], return_series=True,
                                                     heka_dict=heka_dict)
        v = shift_v_rest(v_mat[0], v_rest_shift)
        t = t_mat[0]
        v_traces.append(v)
        t_traces.append(t)
        series_traces.append(series)
    hyper_depo_params_cell = hyper_depo_params[hyper_depo_params['cell_id'] == cell_id]
    amps_hyper_cell = get_array_from_str(hyper_depo_params_cell['amps_hyper'].values[0])
    series_hyper_cell = get_array_from_str(hyper_depo_params_cell['series_hyper'].values[0])
    amps_depo_cell = get_array_from_str(hyper_depo_params_cell['amps_depo'].values[0])
    series_depo_cell = get_array_from_str(hyper_depo_params_cell['series_depo'].values[0])
    series_idxs_hyper = find_series_idx(series_hyper_cell, series_traces)
    series_idxs_depo = find_series_idx(series_depo_cell, series_traces)
    series_idxs = np.concatenate((series_idxs_hyper, series_idxs_depo))
    v_traces = np.array(v_traces, dtype=object)[series_idxs]
    t_traces = np.array(t_traces, dtype=object)[series_idxs]
    amps = np.concatenate((amps_hyper_cell, amps_depo_cell))

    # sort
    sort_idxs = np.argsort(amps)
    amps = amps[sort_idxs]
    v_traces = v_traces[sort_idxs]
    t_traces = t_traces[sort_idxs]

    return amps, v_traces, t_traces


def get_spike_characteristics_and_vstep(v_traces, t_traces, spike_characteristic_params, return_characteristics,
                                        ramp_start, step_start):
    spike_characteristics_mat = init_nan((len(v_traces), len(return_characteristics)))
    v_step = init_nan(len(v_traces))
    for i, (v, t) in enumerate(zip(v_traces, t_traces)):
        onset_idxs_after_ramp = get_AP_onset_idxs(v[to_idx(ramp_start, t[1] - t[0]):to_idx(ramp_start + 10, t[1] - t[0])],
                                                 spike_characteristic_params['AP_threshold'])
        onset_idxs_all = get_AP_onset_idxs(v, spike_characteristic_params['AP_threshold'])

        if len(onset_idxs_after_ramp) >= 1 and len(onset_idxs_all) - len(onset_idxs_after_ramp) == 0:
            v_step[i] = np.mean(v[to_idx(step_start + (ramp_start-step_start)/2, t[1] - t[0]): to_idx(ramp_start, t[1] - t[0])])
            v_rest = np.mean(v[0:to_idx(step_start, t[1] - t[0])])
            std_idx_times = (0, 10)  # rather short so that no global changes interfere
            spike_characteristics_mat[i, :] = get_spike_characteristics(np.array(v, dtype=float),
                                                                        np.array(t, dtype=float),
                                                                        return_characteristics,
                                                                        v_rest=v_rest, std_idx_times=std_idx_times,
                                                                        check=False,
                                                                        **spike_characteristic_params)
            # set to nan if spike on DAP
            if spike_characteristics_mat[i, np.array(return_characteristics) == 'DAP_amp'] > 50:
                spike_characteristics_mat[i, :] = init_nan(len(return_characteristics))
    return spike_characteristics_mat, v_step


if __name__ == '__main__':
    save_dir = '../plots/hyper_depo'
    save_dir_summary = os.path.join(save_dir, 'summary')
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    animal = 'rat'
    protocol_hyper = 'hyperRampTester'
    protocol_depo = 'depoRampTester'
    v_rest_shift = -16
    AP_threshold = 0
    cell_ids = ['2013_12_10b', '2013_12_10c', '2013_12_10d', '2013_12_10h',
                '2013_12_11a', '2013_12_11b', '2013_12_11c', '2013_12_11e', '2013_12_11f',
                '2013_12_12b', '2013_12_12d', '2013_12_12e',
                '2013_12_13b', '2013_12_13c', '2013_12_13d', '2013_12_13e', '2013_12_13f']
    cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)

    hyper_depo_params = pd.read_csv('/home/cf/Phd/DAP-Project/cell_data/hyper_depo_params.csv')

    spike_characteristic_params = {'AP_threshold': 0, 'AP_interval': 2.5, 'AP_width_before_onset': 2,
                                   'fAHP_interval': 4.0, 'DAP_interval': 10, 'order_fAHP_min': 1.0,
                                   'order_DAP_max': 1.0, 'min_dist_to_DAP_max': 0.5, 'k_splines': 3, 's_splines': None}
    return_characteristics = ['DAP_amp', 'DAP_deflection', 'DAP_width', 'fAHP_amp', 'DAP_time']
    ramp_start = 600
    step_start = 200

    spike_characteristic_mat_per_cell = []
    amps_per_cell = []
    v_step_per_cell = []
    for cell_idx, cell_id in enumerate(cell_ids):
        amps, v_traces, t_traces = get_amps_v_traces_t_traces(cell_id, data_dir, protocol_hyper, protocol_depo,
                                                              hyper_depo_params)
        spike_characteristic_mat, v_step = get_spike_characteristics_and_vstep(v_traces, t_traces, spike_characteristic_params,
                                                                               return_characteristics, ramp_start, step_start)
        spike_characteristic_mat_per_cell.append(spike_characteristic_mat)
        amps_per_cell.append(amps)
        v_step_per_cell.append(v_step)

        # save and plot
        save_dir_img = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)
        print cell_id

        c_map = pl.cm.get_cmap('plasma')
        colors = c_map(np.linspace(0, 1, len(v_traces)))

        pl.figure()
        for i, (v, t) in enumerate(zip(v_traces, t_traces)):
            pl.plot(t, v, color=colors[i], label=str(np.round(amps[i], 2)) + ' nA', linewidth=1)
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.legend(fontsize=10)
        pl.xlim(100, 800)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'v.png'))
        #pl.show()

        pl.figure()
        for i, (v, t) in enumerate(zip(v_traces, t_traces)):
            pl.plot(t, v, color=colors[i], label=str(np.round(amps[i], 2)) + ' nA', linewidth=1)
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.legend(fontsize=10)
        pl.xlim(595, 645)
        pl.ylim(-95, -40)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'v_zoom.png'))
        #pl.show()
        pl.close()

    if not os.path.exists(save_dir_summary):
        os.makedirs(save_dir_summary)
    np.save(os.path.join(save_dir_summary, 'spike_characteristic_mat_per_cell.npy'),
            np.array(spike_characteristic_mat_per_cell))
    np.save(os.path.join(save_dir_summary, 'return_characteristics.npy'),
            np.array(return_characteristics))
    np.save(os.path.join(save_dir_summary, 'cell_ids.npy'), np.array(cell_ids))
    np.save(os.path.join(save_dir_summary, 'amps_per_cell.npy'), np.array(amps_per_cell))
    np.save(os.path.join(save_dir_summary, 'v_step_per_cell.npy'), np.array(v_step_per_cell))