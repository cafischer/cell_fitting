from __future__ import division
from heka_reader import HekaReader
import os
import matplotlib.pyplot as pl
import numpy as np
from cell_characteristics.analyze_APs import get_AP_onsets, get_AP_max


def correct_baseline(y, vrest=None, v_rest_change=None):
    if vrest is not None:
        y = y - (y[0] - vrest)
    if v_rest_change is not None:
        y += v_rest_change
    return y


def get_AP_peak(vm, window_before, window_after, threshold, AP_interval):
    AP_peak = None
    AP_onsets = get_AP_onsets(vm, threshold)
    if len(AP_onsets) == 1:
        onset = AP_onsets[0]
        if window_before < onset < len(vm) - window_after:
            AP_peak = get_AP_max(vm, onset, len(vm), interval=AP_interval)
    return AP_peak


if __name__ == '__main__':

    data_dir = '/home/cf/Phd/DAP-Project/cell_data/rawData'
    protocol = 'rampIV'
    v_rest = -75
    correct_vrest = True
    dt = 0.01
    threshold = 20
    window_before_t = 3
    window_after_t = 50
    window_before = int(round(window_before_t / dt))
    window_after = int(round(window_after_t / dt))
    AP_interval = 1 / dt

    cells = os.listdir(data_dir)
    AP_peak_per_cell = []
    cells_with_AP = []
    vm_with_AP = []

    for cell in cells:
        hekareader = HekaReader(os.path.join(data_dir, cell))
        type_to_index = hekareader.get_type_to_index()
        group = 'Group1'
        trace = 'Trace1'
        protocol_to_series = hekareader.get_protocol(group)
        if not protocol in protocol_to_series.keys():
            continue
        series = protocol_to_series[protocol]
        sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series])+1)]
        sweep_idx = range(len(sweeps))
        sweeps = [sweeps[index] for index in sweep_idx]
        indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]

        AP_peak = []
        for index in indices:
            # take next sweep
            t, vm = hekareader.get_xy(index)
            t *= 1000  # ms
            assert dt == t[1] - t[0]  # ms
            vm *= 1000  # mV
            if correct_vrest:
                vm = correct_baseline(vm, v_rest)

            # get AP peak
            AP_peak = get_AP_peak(vm, window_before, window_after, threshold, AP_interval)
            if AP_peak is not None:
                cells_with_AP.append(cell)
                AP_peak_per_cell.append(AP_peak)
                vm_with_AP.append(vm)
                break

    AP_matrix = np.zeros((len(cells_with_AP), window_before + window_after))
    for i, AP_peak in enumerate(AP_peak_per_cell):
        # pl.figure()
        # pl.plot(t[AP_peak - window_before:AP_peak + window_after], vm_with_AP[i][AP_peak - window_before:AP_peak + window_after])
        # pl.plot(AP_peak*dt, vm_with_AP[i][AP_peak], 'or')
        # pl.show()
        AP_matrix[i, :] = vm_with_AP[i][AP_peak - window_before:AP_peak + window_after]

    # plot
    pl.figure()
    for i in range(len(cells_with_AP)):
        pl.plot(np.arange(0, (window_before+window_after)*dt, dt), AP_matrix[i, :])
    pl.show()
