from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell
import os
import json
from cell_fitting.optimization.evaluation.sine_stimulus import apply_sine_stimulus
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/4'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # apply stim
    amp2 = 0.1  # 0.1 # 0.2
    sine1_dur = 10000  # 1000 # 2000 # 5000  # 10000
    freq2 = 1  #1  # 5  # 20
    onset_dur = 500
    offset_dur = 500
    dt = 0.01

    amp1s = [0.5] #np.arange(0.1, 0.55, 0.1)

    sine_params = {'amp2': amp2, 'sine1_dur': sine1_dur, 'freq2': freq2, 'onset_dur': onset_dur,
                   'offset_dur': offset_dur, 'dt': dt}

    # plot
    # save_dir_img = os.path.join(save_dir, 'img', 'sine_stimulus', 'pdfs', 'x_'+str(amp2)+'_'+str(sine1_dur)+'_'+str(freq2))
    # if not os.path.exists(save_dir_img):
    #     os.makedirs(save_dir_img)

    fig, ax = pl.subplots(len(amp1s)+1, 1, sharex=True, figsize=(21, 29.7))
    for i, amp1 in enumerate(amp1s):
        sine_params['amp1'] = amp1
        v, t, i_inj = apply_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)
        if i == 0:
            ax[0].plot(t, i_inj)
        ax[i+1].plot(t, v, 'r')
        ax[i+1].set_ylim(-90, 60)
        ax[i+1].set_title('sine1_dur: %.2f, freq2: %.2f, amp1: %.2f, amp2: %.2f' % (sine1_dur, freq2, amp1, amp2),
                          fontsize=14)
    fig.text(0.06, 0.5, 'Membrane Potential (mV)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.06, 'Time (ms)', ha='center', fontsize=14)
    #pl.savefig(os.path.join(save_dir_img, 'double_sine.pdf'))
    pl.show()
