from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.data import shift_v_rest
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_protocols_same_base
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_characteristics import to_idx
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_fitting.util import init_nan
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    save_dir = './plots/hyper_depo'
    save_dir_summary = os.path.join(save_dir, 'summary')
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    animal = 'rat'
    protocol_hyper = 'hyperRampTester'
    protocol_depo = 'depoRampTester'
    v_rest_shift = -16
    AP_threshold = -30
    repetition = 0
    #cells = get_cells_for_protocol(data_dir, protocol)

    cells = ['2013_12_11e', '2013_12_10c', '2013_12_10b', '2013_12_11b', '2013_12_13f', '2013_12_11a', '2013_12_11c',
             '2013_12_10d', '2013_12_13c', '2013_12_11f', '2013_12_13b']

    #cells_try_other_run = ['2013_12_12e']
    #cells_not_so_good = ['2013_12_12d', '2013_12_13e', '2013_12_13d','2013_12_12b', '2013_12_10h']

    AP_threshold = -30  # mV
    AP_interval = 2.5  # ms (also used as interval for fAHP)
    AP_width_before_onset = 2  # ms
    fAHP_interval = 4.0  # ms
    DAP_interval = 10  # ms
    order_fAHP_min = 1.0  # ms (how many points to consider for the minimum)
    order_DAP_max = 1.0  # ms (how many points to consider for the minimum)
    min_dist_to_DAP_max = 0.5  # ms
    k_splines = 3
    s_splines = None

    amps_hyper = [-0.2, -0.15, -0.1, -0.05]  # must be in steps of 0.05!
    amps_depo = [0.05, 0.1, 0.15, 0.2]
    amps = np.concatenate((amps_hyper, amps_depo))
    DAP_amps_all_cells = init_nan((len(cells), len(amps)))
    DAP_deflections_all_cells = np.zeros((len(cells), len(amps)))
    DAP_widths_all_cells = np.zeros((len(cells), len(amps)))

    for cell_idx, cell_id in enumerate(cells):
        if '2012' in cell_id:
            continue
        if not check_rat_or_gerbil(cell_id) == animal:
            print 'Cell %d not rat!' % cell_idx
            continue
        v_traces = []
        t_traces = []
        DAP_amps = np.zeros(len(amps_hyper)+len(amps_depo))
        DAP_deflections = np.zeros(len(amps_hyper)+len(amps_depo))
        DAP_widths = np.zeros(len(amps_hyper)+len(amps_depo))
        for i, amp in enumerate(amps_hyper):
            idx = int(round((amp + 0.05) / -0.05))
            print amp
            protocol = protocol_hyper if amp == -0.05 else protocol_hyper + '(%d)' % idx
            try:
                v_mat, t_mat = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                     sweep_idxs=[repetition])
                v = shift_v_rest(v_mat[repetition], v_rest_shift)
                t = t_mat[repetition]
                v_traces.append(v)
                t_traces.append(t)

                v_rest = np.mean(v[0:to_idx(100, t[1] - t[0])])
                std_idx_times = (0, 100)
                DAP_amps[i], DAP_deflections[i], DAP_widths[i] = get_spike_characteristics(v, t,
                                                                                           ['DAP_amp', 'DAP_deflection',
                                                                                            'DAP_width'],
                                                                                           v_rest, AP_threshold,
                                                                                           AP_interval,
                                                                                           AP_width_before_onset,
                                                                                           fAHP_interval,
                                                                                           std_idx_times, k_splines,
                                                                                           s_splines,
                                                                                           order_fAHP_min, DAP_interval,
                                                                                           order_DAP_max,
                                                                                           min_dist_to_DAP_max,
                                                                                           check=False)
            except KeyError:
                v_traces.append(np.nan)
                t_traces.append(np.nan)
                DAP_amps[i] = DAP_deflections[i] = DAP_widths[i] = np.nan

        for i, amp in enumerate(amps_depo):
            idx = int(round((amp - 0.05) / 0.05))
            print amp
            protocol = protocol_depo if amp == 0.05 else protocol_depo + '(%d)' % idx
            try:
                v_mat, t_mat = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                     sweep_idxs=[repetition])

                v = shift_v_rest(v_mat[repetition], v_rest_shift)
                t = t_mat[repetition]
                v_traces.append(v)
                t_traces.append(t)

                v_rest = np.mean(v[0:to_idx(100, t[1] - t[0])])
                std_idx_times = (0, 100)
                DAP_amps[i+len(amps_hyper)], DAP_deflections[i+len(amps_hyper)], DAP_widths[i+len(amps_hyper)] \
                                                                               = get_spike_characteristics(v, t,
                                                                               ['DAP_amp', 'DAP_deflection', 'DAP_width'],
                                                                               v_rest, AP_threshold, AP_interval,
                                                                               AP_width_before_onset, fAHP_interval,
                                                                               std_idx_times, k_splines, s_splines,
                                                                               order_fAHP_min, DAP_interval,
                                                                               order_DAP_max, min_dist_to_DAP_max,
                                                                               check=False)
            except KeyError:
                v_traces.append(np.nan)
                t_traces.append(np.nan)
                DAP_amps[i+len(amps_hyper)] = DAP_deflections[i+len(amps_hyper)] = DAP_widths[i+len(amps_hyper)] = np.nan

        # sort
        idx_sort = np.argsort(amps)
        amps = amps[idx_sort]
        DAP_amps = DAP_amps[idx_sort]
        DAP_deflections = DAP_deflections[idx_sort]
        DAP_widths = DAP_widths[idx_sort]
        v_traces = np.array(v_traces)[idx_sort]
        t_traces = np.array(t_traces)[idx_sort]

        # take out real spikes
        DAP_deflections[DAP_amps > 50] = np.nan
        DAP_widths[DAP_amps > 50] = np.nan
        DAP_amps[DAP_amps > 50] = np.nan

        # for all cells
        DAP_amps_all_cells[cell_idx, :] = DAP_amps
        DAP_deflections_all_cells[cell_idx, :] = DAP_deflections
        DAP_widths_all_cells[cell_idx, :] = DAP_widths

        # plot
        save_dir_img = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        c_map = pl.cm.get_cmap('plasma')
        colors = c_map(np.linspace(0, 1, len(v_traces)))

        print cell_id
        pl.figure(figsize=(8, 6))
        for i, (v, t) in enumerate(zip(v_traces, t_traces)):
            pl.plot(t, v, color=colors[i], label=str(np.round(amps[i], 2)) + ' nA')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane potential (mV)')
        pl.legend(loc='upper right')
        pl.tight_layout()
        pl.xlim(0, t_traces[0][-1])
        pl.savefig(os.path.join(save_dir_img, 'v.png'))
        #pl.show()

        pl.figure(figsize=(8, 6))
        for i, (v, t) in enumerate(zip(v_traces, t_traces)):
            pl.plot(t, v, color=colors[i], label=str(np.round(amps[i], 2)) + ' nA')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane potential (mV)')
        pl.legend()
        pl.xlim(595, 645)
        pl.ylim(-95, -40)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'v_zoom.png'))
        #pl.show()

        not_nan = ~np.isnan(DAP_amps)
        pl.figure()
        pl.plot(np.array(amps)[not_nan], np.array(DAP_amps)[not_nan], 'ok')
        pl.xlabel('Current Amplitude (nA)')
        pl.ylabel('DAP Amplitude (mV)')
        pl.xticks(amps)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'DAP_amp.png'))
        #pl.show()

        not_nan = ~np.isnan(DAP_deflections)
        pl.figure()
        pl.plot(np.array(amps)[not_nan], np.array(DAP_deflections)[not_nan], 'ok')
        pl.xlabel('Current Amplitude (nA)')
        pl.ylabel('DAP Deflection (mV)')
        pl.xticks(amps)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'DAP_deflection.png'))
        #pl.show()

        not_nan = ~np.isnan(DAP_widths)
        pl.figure()
        pl.plot(np.array(amps)[not_nan], np.array(DAP_widths)[not_nan], 'ok')
        pl.xlabel('Current Amplitude (nA)')
        pl.ylabel('DAP Width (ms)')
        pl.xticks(amps)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'DAP_width.png'))
        #pl.show()

    if not os.path.exists(save_dir_summary):
        os.makedirs(save_dir_summary)
    np.save(os.path.join(save_dir_summary, 'DAP_amps.npy'), DAP_amps_all_cells)
    np.save(os.path.join(save_dir_summary, 'DAP_deflections.npy'), DAP_deflections_all_cells)
    np.save(os.path.join(save_dir_summary, 'DAP_widths.npy'), DAP_widths_all_cells)
    np.save(os.path.join(save_dir_summary, 'cells.npy'), np.array(cells))
    np.save(os.path.join(save_dir_summary, 'amps.npy'), np.array(amps))