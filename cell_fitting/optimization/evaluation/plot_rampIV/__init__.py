from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.read_heka import get_sweep_index_for_amp, get_i_inj_from_function, get_v_and_t_from_heka, shift_v_rest
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict, plot_v
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.data.plot_rampIV import find_current_threshold_data
pl.style.use('paper')


def evaluate_rampIV(pdf, cell, data_dir, data_dir_characteristics, save_dir):
    save_dir_img = os.path.join(save_dir, 'img', 'rampIV')

    # simulate / load
    ramp_amp = find_current_threshold(cell)
    v, t, i_inj = simulate_rampIV(cell, ramp_amp, v_init=-75, celsius=35)

    v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(data_dir, 'rampIV', return_sweep_idxs=True)
    i_inj_mat = get_i_inj_from_function('rampIV', sweep_idxs, t_mat[0][-1], t_mat[0][1] - t_mat[0][0])
    ramp_amp_data, idx = find_current_threshold_data(v_mat, i_inj_mat, AP_threshold=-10)
    v_data, t_data, i_inj_data = shift_v_rest(v_mat[idx], -16), t_mat[idx], i_inj_mat[idx]
    dt = t_data[1] - t_data[0]

    # evaluate
    rmse, rmse_dap = get_rmse(v, v_data, t_data, i_inj_data, dt)
    return_characteristics = ['DAP_amp', 'DAP_deflection', 'DAP_width', 'DAP_time']
    characteristics_mat_models = get_spike_characteristics(v, t,
                                                    ['DAP_amp', 'DAP_deflection', 'DAP_width', 'DAP_time'],
                                                    v_rest=np.mean(v_data[:np.nonzero(i_inj_data)[0][0]]),
                                                    std_idx_times=(0, 5),
                                                    **get_spike_characteristics_dict())

    # plot in pdf
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)
    np.savetxt(os.path.join(save_dir_img, 'current_threshold.txt'), np.array([ramp_amp]))

    plot_v(t, v, 'r', os.path.join(save_dir_img, '%.2f(nA)' % ramp_amp))
    fig = plot_rampIV_with_data(t, v, t_data, v_data, os.path.join(save_dir_img, '%.2f(nA)' % ramp_amp))
    pdf.savefig(fig)
    pl.close()

    fig = plot_txt('RMSE: %.2f mV' % rmse)
    pdf.savefig(fig)
    pl.close()
    fig = plot_txt('RMSE DAP: %.2f mV' % rmse_dap)
    pdf.savefig(fig)
    pl.close()

    fig = plot_txt('Current Threshold: %.2f mV' % ramp_amp)
    pdf.savefig(fig)
    pl.close()
    np.savetxt(os.path.join(save_dir_img, 'current_threshold.txt'), np.array([ramp_amp]))

    fig = plot_characteristics(characteristics_mat_models, data_dir_characteristics, return_characteristics,
                               save_dir_img)
    pdf.savefig(fig)
    pl.close()


def find_current_threshold(cell):
    for ramp_amp in np.arange(0.5, 4.0+0.1, 0.1):
        v, t, i_inj = simulate_rampIV(cell, ramp_amp)
        start = np.where(i_inj)[0][0]
        onset_idxs = get_AP_onset_idxs(v[start:], threshold=0)
        if len(onset_idxs) >= 1:
            return ramp_amp
    return None


def simulate_rampIV(cell, ramp_amp, v_init=-75, celsius=35, dt=0.01, tstop=161.99, onset=200):
    protocol = 'rampIV'
    sweep_idx = get_sweep_index_for_amp(ramp_amp, protocol)
    i_inj = get_i_inj_from_function(protocol, [sweep_idx], tstop, dt)[0]

    simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': v_init, 'tstop': tstop,
                         'dt': dt, 'celsius': celsius, 'onset': onset}

    v, t, _ = iclamp_handling_onset(cell, **simulation_params)
    return v, t, i_inj


def load_rampIV_data(data_dir, ramp_amp, v_shift=-16):
    sweep_idxs = [get_sweep_index_for_amp(ramp_amp, 'rampIV')]
    v, t = get_v_and_t_from_heka(data_dir, 'rampIV', sweep_idxs=sweep_idxs)
    v = shift_v_rest(v[0], v_shift)
    t = t[0]
    i_inj = get_i_inj_from_function('rampIV', sweep_idxs, t[-1], t[1]-t[0])[0]
    return v, t, i_inj


def plot_characteristics(characteristics_mat_models, data_dir_characteristics, return_characteristics,
                         save_dir_img=None):
    fig, ((ax1, ax2), (ax3, ax4)) = pl.subplots(nrows=2, ncols=2, figsize=(2 * 6.4, 2 * 4.8))
    plot_characteristics_on_axes([ax1, ax2, ax3, ax4], return_characteristics, np.array([characteristics_mat_models], dtype=float),
                                 data_dir_characteristics)
    if save_dir_img is not None:
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)
        pl.savefig(os.path.join(save_dir_img, 'hist.png'))
    return fig


def plot_txt(txt):
    fig = pl.figure(figsize=(10, 0.5))
    pl.text(0.0, 0.0, txt, fontsize=18)
    pl.axis('off')
    pl.subplots_adjust(bottom=0.3, left=0.0)
    return fig


def get_rmse(v_model, v_data, t_data, i_inj_data, dt):
    rmse = np.sqrt(np.mean((v_model - v_data) ** 2))

    fAHP_min_idx, DAP_width_idx = get_spike_characteristics(v_data, t_data, ['fAHP_min_idx', 'DAP_width_idx'],
                                         v_rest=np.mean(v_data[:np.nonzero(i_inj_data)[0][0]]),
                                         std_idx_times=(0, 5),
                                         **get_spike_characteristics_dict())
    rmse_dap = np.sqrt(np.mean((v_model[fAHP_min_idx:DAP_width_idx] - v_data[fAHP_min_idx:DAP_width_idx]) ** 2))
    return rmse, rmse_dap


def plot_rampIV_with_data(t, v, t_data, v_data, save_dir_img=None):
    fig = pl.figure()
    pl.plot(t_data, v_data, 'k', label='Exp. Cell')
    pl.plot(t, v, 'r', label='Model')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.legend()
    pl.tight_layout()
    if save_dir_img is not None:
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)
        pl.savefig(os.path.join(save_dir_img, 'v_model_and_data.png'))
    return fig


def plot_rampIV(t, v, save_dir_img=None):
    fig = pl.figure()
    pl.plot(t, v, 'r', label='Model')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.tight_layout()
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'v.png'))
    return fig


def plot_characteristics_on_axes(ax, return_characteristics, characteristics_mat_models, data_dir):

    characteristics_mat_data = np.load(os.path.join(data_dir, 'characteristics_mat.npy'))
    return_characteristics_data = np.load(os.path.join(data_dir, 'return_characteristics.npy')).tolist()
    idxs = np.array([return_characteristics_data.index(c) for c in return_characteristics])
    characteristics_mat_data = characteristics_mat_data[:, idxs]

    for i, characteristic in enumerate(return_characteristics):
        min_val = np.nanmin([np.nanmin(characteristics_mat_data[:, i]), np.nanmin(characteristics_mat_models[:, i])])
        max_val = np.nanmax([np.nanmax(characteristics_mat_data[:, i]), np.nanmax(characteristics_mat_models[:, i])])
        bins = np.linspace(min_val, max_val, 100)
        if return_characteristics[i] == 'AP_width':
            bins = np.arange(min_val, max_val + 0.015, 0.015)

        hist, bins = np.histogram(characteristics_mat_data[~np.isnan(characteristics_mat_data[:, i]), i], bins=bins)
        hist_models, _ = np.histogram(characteristics_mat_models[~np.isnan(characteristics_mat_models[:, i]), i],
                                      bins=bins)
        character_name_dict = {'AP_amp': 'AP Amplitude (mV)', 'AP_width': 'AP Width (ms)',
                               'fAHP_amp': 'fAHP Amplitude (mV)',
                               'DAP_amp': 'DAP Amplitude (mV)', 'DAP_deflection': 'DAP Deflection (mV)',
                               'DAP_width': 'DAP Width (ms)', 'DAP_time': 'DAP Time (ms)',
                               'fAHP2DAP_time': '$Time_{DAP_{max} - fAHP_{min}} (ms)$'}

        # plot
        ax[i].bar(bins[:-1], hist / np.shape(characteristics_mat_data)[0], width=bins[1] - bins[0], color='k', alpha=0.5)
        ax[i].set_ylim(0, 0.1)
        ax_twin = ax[i].twinx()
        ax_twin.spines['right'].set_visible(True)
        ax_twin.bar(bins[:-1], hist_models, width=bins[1] - bins[0], color='r', alpha=0.5)
        ax_twin.set_ylim(0, 1)
        ax[i].set_xlabel(character_name_dict.get(return_characteristics[i], return_characteristics[i]))
        ax[i].set_ylabel('Proportion Exp. Cells')
        ax_twin.set_ylabel('Proportion Models', fontsize=18, color='r')
        pl.tight_layout()