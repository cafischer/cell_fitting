from cell_fitting.data.data_sinus_mat.read_sinus_mat import get_sinus_data_from_mat
import os
import numpy as np
from cell_characteristics.analyze_APs import get_AP_onset_idxs
import matplotlib.pyplot as pl
pl.style.use('paper')


save_dir = './sinus_mat_files'
repetition = 0
AP_threshold = -30

for file_name in os.listdir(save_dir):
    v, t, i_inj, t_i_inj, amp1, amp2, freq1, freq2 = get_sinus_data_from_mat(os.path.join(save_dir, file_name))

    if np.any(amp2 == 0):
        print file_name

        for i in np.where(amp2 == 0)[0]:

            onsets = get_AP_onset_idxs(v[0, i][:, repetition], AP_threshold)
            n_APs_up = len(onsets[onsets < len(t) / 2])
            n_APs_down = len(onsets[onsets > len(t) / 2])

            print 'Amp: ' + str(amp1[i])
            print n_APs_up - n_APs_down

            pl.figure()
            pl.title('amp1: ' + str(amp1[i]) + ' ' + 'amp2: ' + str(amp2[i]) + ' ' +
                     'freq1: ' + str(freq1[i]) + ' ' + 'freq2: ' + str(freq2[i]) + ' ')
            pl.plot(t[0, i][:, repetition], v[0, i][:, repetition])
            pl.show()


# no cell with hysteresis effect, but maybe amp not small enough or stim too long or noise facilitated switching to spiking state