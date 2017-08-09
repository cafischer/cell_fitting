from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from cell_characteristics.analyze_APs import get_AP_onsets
from analyze_intracellular.load_files import read_channel_data


data_patch = pd.read_csv('../data/2015_08_26b/vrest-75/IV/0.4(nA).csv')
AP_onsets = get_AP_onsets(data_patch.v, threshold=-55)
AP_onset = AP_onsets[3]
AP_end = AP_onsets[4]
v_spike = data_patch.v[AP_onset]
t_spike = data_patch.t[AP_onset]
t_patch = np.array(data_patch.t)[AP_onset:AP_end] - np.array(data_patch.t)[AP_onset]
v_patch = np.array(data_patch.v)[AP_onset:AP_end]

data_SH = pd.read_csv('/home/cf/Phd/DAP-Project/paper/grid cells/input/Schmidt-Hieber-2013/DAP_digitized.csv', header=None)
t_SH = data_SH.iloc[1:, 0] + np.abs(data_SH.iloc[1, 0])
v_SH = data_SH.iloc[1:, 1] + np.abs(data_SH.iloc[1, 1]) + v_spike - 2

data_dir = '/home/cf/Phd/programming/projects/analyze_intracellular/analyze_intracellular/data/kazu_data_raw/4714/4714-001/4714-001.dat'
v_sharp = read_channel_data(data_dir, channel=10, n_channels=10) / 100 + 51.5
t_sharp = np.arange(0, len(v_sharp)) * 1.0 / 20000 * 1000
i_sharp = read_channel_data(data_dir, channel=9, n_channels=10)
threshold_pos = 750
i_on_pos = np.nonzero(np.diff(np.sign(i_sharp - threshold_pos)) == 2)[0]
i_off_pos = np.nonzero(np.diff(np.sign(threshold_pos - i_sharp)) == 2)[0]
i_idx = 2
AP_onsets = get_AP_onsets(v_sharp[i_on_pos[i_idx]:i_off_pos[i_idx]], threshold=-33.5)
AP_onset = AP_onsets[1] + i_on_pos[i_idx]
AP_end = AP_onsets[2] + i_on_pos[i_idx]

pl.figure()
pl.plot(t_patch, v_patch, label='patch in vitro')
pl.plot(t_SH, v_SH, label='patch in vivo')
pl.plot(t_sharp[AP_onset:AP_end] - t_sharp[AP_onset], v_sharp[AP_onset:AP_end], label='sharp in vivo')
pl.xlim(0, 50)
pl.ylim(-70, -46)
pl.legend()
pl.show()