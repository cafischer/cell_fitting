import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.data import shift_v_rest
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_protocols_same_base
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_characteristics import to_idx
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    save_dir = './plots/hyperdap'
    save_dir_summary = os.path.join(save_dir, 'summary_plots')
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'hyperRampTester'
    protocol2 = 'depoRampTester'
    v_rest_shift = -16
    AP_threshold = -30
    repetition = 0
    amps_test = np.array([-0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2])
    #cells = get_cells_for_protocol(data_dir, protocol)

    cells = ['2013_12_11b', '2013_12_13f', '2013_12_11a', '2013_12_10b', '2013_12_11c',
             '2013_12_11e', '2013_12_10d', '2013_12_10c', '2013_12_13c', '2013_12_11f', '2013_12_13b']

    for cell in cells:
        print check_rat_or_gerbil(cell)

    #cells_try_other_run = ['2013_12_12e']
    #cells_not_so_good = ['2013_12_12d', '2013_12_13e', '2013_12_13d','2013_12_12b', '2013_12_10h']

    AP_threshold = -30  # mV
    AP_interval = 2.5  # ms (also used as interval for fAHP)
    AP_width_before_onset = 2  # ms
    DAP_interval = 10  # ms
    order_fAHP_min = 1.0  # ms (how many points to consider for the minimum)
    order_DAP_max = 1.0  # ms (how many points to consider for the minimum)
    min_dist_to_DAP_max = 0.5  # ms
    k_splines = 3
    s_splines = None

    DAP_amps_all_cells = np.zeros((len(cells), 8))
    DAP_deflections_all_cells = np.zeros((len(cells), 8))

    for cell_idx, cell in enumerate(cells):
        if '2012' in cell:
            continue

        protocols = get_protocols_same_base(os.path.join(data_dir, cell+'.dat'), protocol)
        vs = []
        ts = []
        amps = []
        DAP_amps = []
        DAP_deflections = []
        for i, p in enumerate(protocols):
            if -0.05 + i * -0.05 < -0.2:
                break
            v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell+'.dat'), p,
                                                         return_sweep_idxs=True)
            v = shift_v_rest(v_mat[repetition], v_rest_shift)
            t = t_mat[repetition]
            vs.append(v)
            ts.append(t)
            amps.append(-0.05 + i * -0.05)

            v_rest = np.mean(v[0:to_idx(100, t[1] - t[0])])
            std_idx_times = (0, 100)
            DAP_amp, DAP_deflection = get_spike_characteristics(v, t, ['DAP_amp', 'DAP_deflection'],
                                                                v_rest, AP_threshold,
                                                                AP_interval, AP_width_before_onset, std_idx_times,
                                                                k_splines, s_splines, order_fAHP_min, DAP_interval,
                                                                order_DAP_max, min_dist_to_DAP_max, check=False)
            DAP_amps.append(DAP_amp)
            DAP_deflections.append(DAP_deflection)
        vs = vs[::-1]
        ts = ts[::-1]
        amps = amps[::-1]
        DAP_amps = DAP_amps[::-1]
        DAP_deflections = DAP_deflections[::-1]

        protocols = get_protocols_same_base(os.path.join(data_dir, cell+'.dat'), protocol2)
        for i, p in enumerate(protocols):
            if 0.05 + i * 0.05 > 0.2:
                break
            v_mat, t_mat = get_v_and_t_from_heka(os.path.join(data_dir, cell+'.dat'), p)
            v = shift_v_rest(v_mat[repetition], v_rest_shift)
            t = t_mat[repetition]
            vs.append(v)
            ts.append(t)
            amps.append(0.05 + i * 0.05)

            v_rest = np.mean(v[0:to_idx(100, t[1] - t[0])])
            std_idx_times = (0, 100)
            DAP_amp, DAP_deflection = get_spike_characteristics(v, t, ['DAP_amp', 'DAP_deflection'],
                                                                v_rest, AP_threshold,
                                                                AP_interval, AP_width_before_onset, std_idx_times,
                                                                k_splines, s_splines, order_fAHP_min, DAP_interval,
                                                                order_DAP_max, min_dist_to_DAP_max, check=False)
            DAP_amps.append(DAP_amp)
            DAP_deflections.append(DAP_deflection)

        traces_there = np.array([amp in np.round(amps, 2) for amp in np.round(amps_test, 2)])
        DAP_amps = np.array(DAP_amps, dtype=float)  # take out real spikes
        DAP_amps[DAP_amps > 50] = np.nan
        DAP_deflections = np.array(DAP_deflections, dtype=float)  # take out real spikes
        DAP_deflections[DAP_amps > 50] = np.nan
        DAP_amps_all_cells[cell_idx, traces_there] = DAP_amps
        DAP_deflections_all_cells[cell_idx, traces_there] = DAP_deflections


        # plot
        save_dir_fig = os.path.join(save_dir, cell)
        if not os.path.exists(save_dir_fig):
            os.makedirs(save_dir_fig)

        c_map = pl.cm.get_cmap('plasma')
        colors = c_map(np.linspace(0, 1, len(vs)))

        print cell

        pl.figure(figsize=(8, 6))
        for i, (v, t) in enumerate(zip(vs, ts)):
            pl.plot(t, v, color=colors[i], label=str(np.round(amps[i], 2)) + ' nA')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane potential (mV)')
        pl.legend(loc='upper right')
        pl.tight_layout()
        pl.xlim(0, t[-1])
        pl.savefig(os.path.join(save_dir_fig, 'v.png'))
        #pl.show()

        pl.figure(figsize=(8, 6))
        for i, (v, t) in enumerate(zip(vs, ts)):
            pl.plot(t, v, color=colors[i], label=str(np.round(amps[i], 2)) + ' nA')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane potential (mV)')
        pl.legend()
        pl.xlim(595, 645)
        pl.ylim(-95, -40)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_fig, 'v_zoom.png'))
        #pl.show()


        not_nan = ~np.isnan(DAP_amps)
        pl.figure()
        pl.plot(np.array(amps)[not_nan], np.array(DAP_amps)[not_nan], 'ok')
        pl.xlabel('Current Amplitude (nA)')
        pl.ylabel('DAP Amplitude (mV)')
        pl.xticks(amps_test)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_fig, 'DAP_amp.png'))
        #pl.show()

        not_nan = ~np.isnan(DAP_deflections)
        pl.figure()
        pl.plot(np.array(amps)[not_nan], np.array(DAP_deflections)[not_nan], 'ok')
        pl.xlabel('Current Amplitude (nA)')
        pl.ylabel('DAP Deflection (mV)')
        pl.xticks(amps_test)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_fig, 'DAP_deflection.png'))
        #pl.show()

    if not os.path.exists(save_dir_summary):
        os.makedirs(save_dir_summary)
    np.save(os.path.join(save_dir_summary, 'DAP_amps.npy'), DAP_amps_all_cells)
    np.save(os.path.join(save_dir_summary, 'DAP_deflections.npy'), DAP_deflections_all_cells)
    np.save(os.path.join(save_dir_summary, 'cells.npy'), np.array(cells))
    np.save(os.path.join(save_dir_summary, 'amps.npy'), np.array(amps))