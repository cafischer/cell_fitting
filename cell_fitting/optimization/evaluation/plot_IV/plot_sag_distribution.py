from __future__ import  division
import os
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import seaborn as sns
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.simulate import iclamp_adaptive_handling_onset
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function, get_sweep_index_for_amp
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import compute_v_sag_and_steady_state

pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'
    mechanism_dir = '../../../model/channels/vavoulis'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'
    save_dir_img = os.path.join(save_dir, 'img', 'IV', 'sag_hist')
    save_dir_data = os.path.join('../../../data/plots/IV/sag_hist', 'rat')
    model_ids = range(1, 7)

    amp = -0.15
    AP_threshold = 0
    sweep_idx = get_sweep_index_for_amp(amp, 'IV')
    load_mechanism_dir(mechanism_dir)

    sag_amps_model = []
    v_deflections_model = []

    for model_id in model_ids:
        # load model
        cell = Cell.from_modeldir(os.path.join(save_dir, str(model_id), 'cell.json'))

        # read data
        v_mat_data, t_mat_data = get_v_and_t_from_heka(data_dir, 'IV', sweep_idxs=[sweep_idx])
        i_inj_mat = get_i_inj_from_function('IV', [sweep_idx], t_mat_data[0][-1], t_mat_data[0][1]-t_mat_data[0][0])
        start_step_idx = np.nonzero(i_inj_mat[0])[0][0]
        end_step_idx = np.nonzero(i_inj_mat[0])[0][-1] + 1

        # plot_IV for model
        simulation_params = {'sec': ('soma', None), 'celsius': 35, 'onset': 200, 'v_init': -75,
                             'tstop': t_mat_data[0, -1], 'dt': t_mat_data[0, 1] - t_mat_data[0, 0],
                             'i_inj': i_inj_mat[0]}
        v_model, t_model, _ = iclamp_adaptive_handling_onset(cell, **simulation_params)

        v_sags, v_steady_states, _ = compute_v_sag_and_steady_state([v_model], [amp], AP_threshold,
                                                                    start_step_idx, end_step_idx)
        sag_amp = v_steady_states[0] - v_sags[0]
        sag_amps_model.append(sag_amp)

        vrest = np.mean(v_model[:start_step_idx])
        v_deflection = vrest - v_steady_states[0]
        v_deflections_model.append(v_deflection)

    # plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    sag_amps_data = np.load(os.path.join(save_dir_data, 'sag_amps.npy'))
    v_deflections_data = np.load(os.path.join(save_dir_data, 'v_deflections.npy'))

    # plot sag amps
    min_val = 0 #min(np.min(sag_amps_data), np.min(sag_amps_model))
    max_val = max(np.max(sag_amps_data), np.max(sag_amps_model))
    bins = np.linspace(min_val, max_val, 100)
    hist_data, bins = np.histogram(sag_amps_data, bins=bins)
    hist_models, _ = np.histogram(sag_amps_model, bins=bins)

    fig, ax1 = pl.subplots()
    #ax1.bar(bins[:-1], hist_data / len(sag_amps_data), width=bins[1] - bins[0], color='k', alpha=0.5)
    ax1.bar(bins[:-1], hist_data / len(sag_amps_data), width=bins[1] - bins[0], color='k', alpha=0.5)
    #ax1.set_ylim(0, 0.1)
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True)
    bars = ax2.bar(bins[:-1], hist_models / len(model_ids), width=bins[1] - bins[0], color='r', alpha=0.5)

    for i, model_id in enumerate(model_ids):
        model_bin = np.digitize(sag_amps_model, bins)
        if np.any(np.isnan(sag_amps_model)):
            model_bin = np.array(model_bin, dtype=float)
            model_bin[np.isnan(sag_amps_model)] = np.nan
        for m_bin in np.unique(model_bin[~np.isnan(model_bin)]):
            h = np.sum(m_bin == model_bin) / len(model_ids)
            s = ''
            model_idxs = np.where(m_bin == model_bin)[0]
            for j, idx in enumerate(model_idxs):
                if j == len(model_idxs) - 1:
                    s += str(model_ids[idx])
                else:
                    s += str(model_ids[idx]) + ','
            dbin = np.diff(bins)[0]
            ax2.annotate(s, xy=(bins[int(m_bin)]-2*dbin, h + 0.01), color='r', fontsize=8)

    ax2.set_ylim(0, 1)
    ax1.set_xlabel('Sag Amplitude (mV)')
    ax1.set_ylabel('Proportion Exp. Cells')
    ax2.set_ylabel('Proportion Models', fontsize=18, color='r')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'hist_sag.png'))
    pl.show()

    # plot deflections
    min_val = 0  # min(np.min(v_deflections_data), np.min(v_deflections_model))
    max_val = max(np.max(v_deflections_data), np.max(v_deflections_model))
    bins = np.linspace(min_val, max_val, 100)
    hist_data, bins = np.histogram(v_deflections_data, bins=bins)
    hist_models, _ = np.histogram(v_deflections_model, bins=bins)

    fig, ax1 = pl.subplots()
    ax1.bar(bins[:-1], hist_data / len(v_deflections_data), width=bins[1] - bins[0], color='k', alpha=0.5)
    ax1.set_ylim(0, 0.2)
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True)
    bars = ax2.bar(bins[:-1], hist_models / len(model_ids), width=bins[1] - bins[0], color='r', alpha=0.5)

    for i, model_id in enumerate(model_ids):
        model_bin = np.digitize(v_deflections_model, bins)
        if np.any(np.isnan(v_deflections_model)):
            model_bin = np.array(model_bin, dtype=float)
            model_bin[np.isnan(v_deflections_model)] = np.nan
        for m_bin in np.unique(model_bin[~np.isnan(model_bin)]):
            h = np.sum(m_bin == model_bin) / len(model_ids)
            s = ''
            model_idxs = np.where(m_bin == model_bin)[0]
            for j, idx in enumerate(model_idxs):
                if j == len(model_idxs) - 1:
                    s += str(model_ids[idx])
                else:
                    s += str(model_ids[idx]) + ','
            dbin = np.diff(bins)[0]
            ax2.annotate(s, xy=(bins[int(m_bin)]-2*dbin, h + 0.01), color='r', fontsize=8)

    ax2.set_ylim(0, 1)
    ax1.set_xlabel('Voltage Deflection (mV)')
    ax1.set_ylabel('Proportion Exp. Cells')
    ax2.set_ylabel('Proportion Models', fontsize=18, color='r')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'hist_deflection.png'))
    
    # joint plot
    data = pd.DataFrame(np.array([sag_amps_data, v_deflections_data]).T, columns=['Sag Amplitude',
                                                                                  'Voltage Deflection'])
    jp = sns.jointplot('Sag Amplitude', 'Voltage Deflection', data=data, stat_func=None, color='k', alpha=0.5)
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = sag_amps_model
    jp.y = v_deflections_model
    jp.plot_joint(pl.scatter, c='r', alpha=0.5)
    for i, model_id in enumerate(model_ids):
        pl.gca().annotate(str(model_id), xy=(sag_amps_model[i] + 0.01, v_deflections_model[i] + 0.4), color='r',
                          fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'sag_amps_deflection_hist.png'))
    pl.show()