from __future__ import division
from heka_reader import HekaReader
import matplotlib.pyplot as pl
import os
import numpy as np
from cell_characteristics.analyze_APs import get_AP_onsets


if __name__ == '__main__':

    cells = ['2013_02_06b', '2013_02_07g', '2013_02_11d', '2013_02_11g', '2013_02_12b']
    data_dir = '/home/cf/Phd/programming/projects/cell_fitting/data'

    for cell in cells:
        save_dir = os.path.join(data_dir, cell, 'analyses')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        hekareader = HekaReader(os.path.join(data_dir, cell, cell+'.dat'))
        type_to_index = hekareader.get_type_to_index()
        group = 'Group1'
        protocol_to_series = hekareader.get_protocol(group)

        protocol = 'Noise1'
        trace = 'Trace1'
        series = protocol_to_series[protocol]
        sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series])+1)]
        sweep_idx = [0]  #range(len(sweeps))
        sweeps = [sweeps[index] for index in sweep_idx][::3]
        indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]

        for i, index in enumerate(indices):
            # get voltage and time
            t, v = hekareader.get_xy(index)
            t *= 1000
            v *= 1000
            dt = t[1] - t[0]

            pl.figure()
            pl.plot(t, v)
            pl.show()

            # get AP_onsets
            AP_onsets = get_AP_onsets(v, -10)

            # ISI hist
            bins = np.arange(0, 1000 + 10, 10)
            ISIs = np.diff(AP_onsets * dt)
            pl.figure()
            pl.hist(ISIs, bins=bins)
            pl.savefig(os.path.join(save_dir, 'ISI_hist.png'))
            pl.show()

            # percent doublet (ISI < 50 ms) and theta (170 ms <= ISI < 330 ms)
            hist, bins = np.histogram(ISIs, bins=bins)
            n_ISI = len(ISIs)
            n_doublets = np.sum(hist[bins[:-1] < 50])
            percent_doublets = n_doublets / n_ISI
            print('percent doublets: ', percent_doublets)
            n_theta = np.sum(hist[np.logical_and(170 <= bins[:-1], bins[:-1] < 330)])
            percent_theta = n_theta / n_ISI
            print('percent theta: ', percent_theta)