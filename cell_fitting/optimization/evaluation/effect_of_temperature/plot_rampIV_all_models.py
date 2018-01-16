from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.evaluation.plot_rampIV import simulate_rampIV
from cell_fitting.optimization.evaluation.effect_of_temperature import set_q10, set_q10_g
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_spike_characteristics
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    model_ids = range(1, 7)
    mechanism_dir = '../../../model/channels/vavoulis_with_temp'
    ramp_amp = 3.0
    q10s = [1.0, 1.5, 2.0, 2.5, 3.0]
    q10_g = 1.0
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'
    rescale = False
    celsius = 22
    temp = 36
    dt_orig = 0.01
    load_mechanism_dir(mechanism_dir)

    # for models
    AP_threshold = -20  # mV
    AP_interval = 8.0  # TODO 2.5  # ms
    fAHP_interval = 15.0  # TODO: 4.0
    AP_width_before_onset = 3.0  # ms
    DAP_interval = 50  # TODO: 10  # ms
    order_fAHP_min = 1.0  # ms (how many points to consider for the minimum)
    order_DAP_max = 1.0  # ms (how many points to consider for the minimum)
    min_dist_to_DAP_max = 0.5  # ms
    k_splines = 3
    s_splines = 0
    std_idx_times = (None, None)
    return_characteristics = ['AP_amp', 'AP_width', 'fAHP_amp', 'DAP_amp', 'DAP_deflection', 'DAP_width', 'DAP_time_abs']

    spike_characteristics_models = np.zeros((len(q10s), len(return_characteristics), len(model_ids)))
    for model_idx, model_id in enumerate(model_ids):
        # load model
        cell = Cell.from_modeldir(os.path.join(save_dir, str(model_id), 'cell.json'))

        v_list = []
        t_list = []
        for q10_idx, q10 in enumerate(q10s):
            set_q10(cell, q10)
            set_q10_g(cell, q10_g)
            v, t, i_inj = simulate_rampIV(cell, ramp_amp, v_init=-75, celsius=celsius, dt=dt_orig, tstop=1000)
            start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
            v_rest = np.mean(v[0:start_i_inj])
            v = v[to_idx(10, dt_orig):]
            dt = dt_orig
            if rescale:
                qt = q10 ** ((celsius - temp) / 10)
                dt *= qt
            t = np.arange(len(v)) * dt
            v_list.append(v)
            t_list.append(t)

            # get spike characteristics
            spike_characteristics_models[q10_idx, :, model_idx] = np.array(get_spike_characteristics(v, t, return_characteristics, v_rest,
                                                                           AP_threshold, AP_interval,
                                                                           AP_width_before_onset,
                                                                           fAHP_interval, std_idx_times,
                                                                           k_splines, s_splines, order_fAHP_min,
                                                                           DAP_interval, order_DAP_max,
                                                                           min_dist_to_DAP_max, round_idxs=True,
                                                                           check=False), dtype=float)

        # plot
        if rescale:
            save_dir_img_model = os.path.join(save_dir, str(model_id), 'img', 'effect_of_temperature', 'rampIV', 'rescaled', 'q10_g_' + str(q10_g))
        else:
            save_dir_img_model = os.path.join(save_dir, str(model_id), 'img', 'effect_of_temperature', 'rampIV', 'not_rescaled', 'q10_g_' + str(q10_g))
        if not os.path.exists(save_dir_img_model):
            os.makedirs(save_dir_img_model)

        # pl.figure()
        # colors = pl.cm.plasma(np.linspace(0, 1, len(q10s)))
        # for i, (v, t) in enumerate(zip(v_list, t_list)):
        #     pl.plot(t, v, color=colors[i], label='q10=%.1f' % q10s[i])
        # pl.legend()
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Membrane Potential (mV)')
        # pl.xlim(0, 150)
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_img_model, 'v_%.2fnA.png' % ramp_amp))
        # #pl.show()
        #
        # pl.figure()
        # colors = pl.cm.plasma(np.linspace(0, 1, len(q10s)))
        # for i, (v, t) in enumerate(zip(v_list, t_list)):
        #     pl.plot(t, v, color=colors[i], label='q10=%.1f' % q10s[i])
        # pl.legend()
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Membrane Potential (mV)')
        # pl.xlim(0, 40)
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_img_model, 'v_%.2fnA_zoom.png' % ramp_amp))
        # #pl.show()


    if rescale:
        save_dir_img_model = os.path.join(save_dir, 'img', 'effect_of_temperature', 'rampIV', 'rescaled', 'q10_g_' + str(q10_g))
    else:
        save_dir_img_model = os.path.join(save_dir, 'img', 'effect_of_temperature', 'rampIV', 'not_rescaled', 'q10_g_' + str(q10_g))
    if not os.path.exists(save_dir_img_model):
        os.makedirs(save_dir_img_model)

    # for characteristic_idx, characteristic_name in enumerate(return_characteristics):
    #     pl.figure()
    #     cm = pl.cm.get_cmap('plasma')
    #     colors = cm(np.linspace(0, 1, len(q10s)))
    #     markers = [(1, 3, 0) if i == 0 else (i+2 % 7, 0, 0) for i in range(len(model_ids))]
    #     for model_idx, model_id in enumerate(model_ids):
    #         for q10_idx, q10 in enumerate(q10s):
    #             pl.plot(q10, spike_characteristics_models[q10_idx, characteristic_idx, model_idx],
    #                     marker=markers[model_idx], color=colors[q10_idx], label=model_id if q10_idx==0 else '')
    #     pl.xlabel('Q10')
    #     pl.ylabel(characteristic_name)
    #     pl.legend(fontsize=12, title='Model')
    #     pl.xticks(q10s, q10s)
    #     pl.xlim(0.5, 4.0)
    #     pl.tight_layout()
    #     pl.savefig(os.path.join(save_dir_img_model, characteristic_name + '.png'))
    # pl.show()

    for characteristic_idx, characteristic_name in enumerate(return_characteristics):
        pl.figure()
        cm = pl.cm.get_cmap('jet')
        colors = cm(np.linspace(0, 1, len(model_ids)))
        for model_idx, model_id in enumerate(model_ids):
            pl.plot(q10s, spike_characteristics_models[:, characteristic_idx, model_idx],
                    color=colors[model_idx], marker='o', label=model_id)
        pl.xlabel('Q10')
        pl.ylabel(characteristic_name)
        pl.legend(fontsize=12, title='Model')
        pl.xticks(q10s, q10s)
        pl.xlim(0.5, 4.0)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img_model, characteristic_name + '.png'))
    #pl.show()