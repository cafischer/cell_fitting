from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell
import os
import json
from cell_fitting.new_optimization.evaluation.sine_stimulus import apply_sine_stimulus
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/1'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # apply stim
    amp1 = 0
    sine1_dur = 2000
    amp2s = np.arange(0.1, 1.1, 0.1)
    freq2s = np.arange(1, 16, 2)
    shifts = [-0.5, 0, 0.5]
    onset_dur = 500
    offset_dur = 500
    dt = 0.01

    for shift in shifts:
        for freq2 in freq2s:
            for amp2 in amp2s:
                sine_params = {'amp1': amp1, 'amp2': amp2, 'sine1_dur': sine1_dur, 'freq2': freq2, 'onset_dur': onset_dur,
                               'offset_dur': offset_dur, 'dt': dt}

                v, t, i_inj = apply_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt,
                                                  shift=shift)

                # plot
                save_dir_img = os.path.join(save_dir, 'img', 'sine_stimulus', 'single_sine', str(sine1_dur))
                if not os.path.exists(save_dir_img):
                    os.makedirs(save_dir_img)

                pl.figure()
                pl.title('freq2: ' + str(freq2) + ' amp2: ' + str(amp2) + ' shift2: '+str(shift), fontsize=16)
                pl.plot(t, v, 'r', linewidth=1.0)
                pl.xlabel('Time (ms)')
                pl.ylabel('Membrane Potential (mV)')
                pl.xlim(onset_dur, t[-1]-offset_dur)
                pl.ylim(-90, 60)
                pl.tight_layout()
                pl.savefig(os.path.join(save_dir_img, 'v_'+ '_shift_'+str(shift) + '_freq_' + str(freq2) + '_amp_' +
                                        str(amp2) + '.png'))
                #pl.show()
                pl.close()