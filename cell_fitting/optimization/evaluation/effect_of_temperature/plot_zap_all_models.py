from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.evaluation.plot_zap import compute_smoothed_impedance, \
    compute_res_freq_and_q_val
from cell_fitting.optimization.evaluation.effect_of_temperature import set_q10, set_q10_g
from cell_fitting.read_heka.i_inj_functions import get_i_inj_zap
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_characteristics.analyze_APs import get_AP_onset_idxs
pl.style.use('paper')


def simulate_zap_with_offset(cell, amp_offset=0, amp=0.1, freq0=0, freq1=20, onset_dur=2000, offset_dur=2000, zap_dur=30000,
                             tstop=34000, dt=0.01, v_init=-75, celsius=35, onset=200):
    i_zap = get_i_inj_zap(amp=amp, freq0=freq0, freq1=freq1, onset_dur=onset_dur, offset_dur=offset_dur,
                          zap_dur=zap_dur, tstop=tstop, dt=dt)
    i_offset = np.ones(len(i_zap)) * amp_offset
    i_exp = i_offset + i_zap
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init, 'tstop': tstop,
                         'dt': dt, 'celsius': celsius, 'onset': onset}
    v, t, i_inj = iclamp_handling_onset(cell, **simulation_params)
    return v, t, i_inj


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    model_ids = range(1, 7)
    mechanism_dir = '../../../model/channels/vavoulis_with_temp'
    q10s = [1.0, 1.5, 2.0, 2.5, 3.0]
    q10_g = 1.0
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'
    rescale = False
    amp_offset = 0.15
    amp_zap = 0.05
    celsius = 22
    temp = 36
    dt_orig = 0.01
    load_mechanism_dir(mechanism_dir)

    freq0 = 0
    freq1 = 20
    onset_dur = 2000
    offset_dur = 2000
    zap_dur = 30000
    tstop = 34000
    dt = 0.01

    res_freq_models = np.zeros((len(q10s), len(model_ids)))
    for model_idx, model_id in enumerate(model_ids):
        # if model_id == 4:
        #     amp = amp_zap - 0.02
        # else:
        #     amp = amp_zap

        # load model
        cell = Cell.from_modeldir(os.path.join(save_dir, str(model_id), 'cell.json'))

        v_list = []
        t_list = []
        for q10_idx, q10 in enumerate(q10s):
            set_q10(cell, q10)
            set_q10_g(cell, q10_g)
            v, t, i_inj = simulate_zap_with_offset(cell, amp_offset=amp_offset, amp=amp_zap)
            dt = dt_orig
            if rescale:
                qt = q10 ** ((celsius - temp) / 10)
                dt *= qt
            t = np.arange(len(v)) * dt
            v_list.append(v)
            t_list.append(t)

            # resonance frequency
            if len(get_AP_onset_idxs(v, -10)) > 0:
                res_freq_models[q10_idx, model_idx] = np.nan
            else:
                imp_smooth, frequencies = compute_smoothed_impedance(v, freq0, freq1, i_inj, offset_dur, onset_dur, tstop, dt)
                res_freq_models[q10_idx, model_idx], _ = compute_res_freq_and_q_val(imp_smooth, frequencies)


        # plot
        if rescale:
            save_dir_img_model = os.path.join(save_dir, str(model_id), 'img', 'effect_of_temperature', 'zap',
                                              'rescaled', 'amp_zap_%.2f_amp_offset_%.2f' % (amp_zap, amp_offset))
        else:
            save_dir_img_model = os.path.join(save_dir, str(model_id), 'img', 'effect_of_temperature', 'zap',
                                              'not_rescaled', 'amp_zap_%.2f_amp_offset_%.2f' % (amp_zap, amp_offset))
        if not os.path.exists(save_dir_img_model):
            os.makedirs(save_dir_img_model)

        pl.figure()
        colors = pl.cm.plasma(np.linspace(0, 1, len(q10s)))
        for i, (v, t) in enumerate(zip(v_list, t_list)):
            pl.plot(t, v, color=colors[i], label='q10=%.1f' % q10s[i])
        pl.legend()
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        #pl.xlim(0, 150)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img_model, 'v.png'))
        #pl.show()


    if rescale:
        save_dir_img = os.path.join(save_dir, 'img', 'effect_of_temperature', 'zap', 'rescaled',
                                    'amp_zap_%.2f_amp_offset_%.2f' % (amp_zap, amp_offset))
    else:
        save_dir_img = os.path.join(save_dir, 'img', 'effect_of_temperature', 'zap', 'not_rescaled',
                                    'amp_zap_%.2f_amp_offset_%.2f' % (amp_zap, amp_offset))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    pl.figure()
    cm = pl.cm.get_cmap('jet')
    colors = cm(np.linspace(0, 1, len(model_ids)))
    for model_idx, model_id in enumerate(model_ids):
        pl.plot(q10s, res_freq_models[:, model_idx],
                color=colors[model_idx], marker='o', label=model_id)
    pl.xlabel('Q10')
    pl.ylabel('Resonance Frequency (Hz)')
    pl.legend(fontsize=12, title='Model')
    pl.xticks(q10s, q10s)
    pl.xlim(0.5, 4.0)
    pl.ylim(0, 10)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'res_freq.png'))
    #pl.show()