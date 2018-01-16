import numpy as np
import matplotlib.pyplot as pl
import os
from cell_fitting.optimization.simulate import iclamp_handling_onset, extract_simulation_params
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_characteristics import to_idx
from cell_fitting.util import merge_dicts
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function, get_sweep_index_for_amp
from cell_characteristics.fIcurve import compute_fIcurve
from cell_fitting.data.plot_IV.plot_fI_curve_fit_distribution import fit_fun
from scipy.optimize import curve_fit
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import compute_v_sag_and_steady_state
from cell_fitting.optimization.evaluation import plot_v, joint_plot_data_and_model
pl.style.use('paper')


def evaluate_IV(pdf, cell, data_dir, data_dir_FI_fit, data_dir_sag, save_dir):
    protocol = 'IV'
    save_dir_img = os.path.join(save_dir, 'img', protocol)

    # simulate / load
    v_mat_data, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir), protocol,
                                                     sweep_idxs=None, return_sweep_idxs=True)
    i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t_mat[0][-1], t_mat[0][1] - t_mat[0][0])

    v_mat_model = list()
    for i in range(len(sweep_idxs)):
        sim_params = {'celsius': 35, 'onset': 200}
        simulation_params = merge_dicts(extract_simulation_params(v_mat_data[i], t_mat[i], i_inj_mat[i]), sim_params)
        v_model, t_model, _ = iclamp_handling_onset(cell, **simulation_params)
        v_mat_model.append(v_model)

    # evaluate
    # fi curve
    amps, firing_rates_data = compute_fIcurve(v_mat_data, i_inj_mat, t_mat[0])
    amps, firing_rates_model = compute_fIcurve(v_mat_model, i_inj_mat, t_mat[0])

    amps, firing_rates_data, firing_rates_model, v_mat_data, v_mat_model = sort_according_to_amps(amps,
                                                                                                  firing_rates_data,
                                                                                                  firing_rates_model,
                                                                                                  v_mat_data,
                                                                                                  v_mat_model)

    amps_greater0_idx = amps >= 0
    amps_greater0, firing_rates_data, firing_rates_model = use_amps_greater0(amps, amps_greater0_idx, firing_rates_data,
                                                                             firing_rates_model)

    # sag
    amp = -0.15
    start_step_idx = np.nonzero(i_inj_mat[0])[0][0]
    end_step_idx = np.nonzero(i_inj_mat[0])[0][-1] + 1
    v_sags, v_steady_states, _ = compute_v_sag_and_steady_state([v_mat_model[0]], [amp], -10,
                                                                start_step_idx, end_step_idx)
    vrest = np.mean(v_mat_model[0][:start_step_idx])
    sag_amp = v_steady_states[0] - v_sags[0]
    v_deflection = vrest - v_steady_states[0]

    sag_amps_data = np.load(os.path.join(data_dir_sag, 'sag_amps.npy'))
    v_deflections_data = np.load(os.path.join(data_dir_sag, 'v_deflections.npy'))

    # fit FI-curve
    b0 = amps_greater0[np.where(firing_rates_model > 0)[0][0]]
    try:
        p_opt, _ = curve_fit(fit_fun, amps_greater0, firing_rates_model, p0=[50, b0, 0.5])
    except RuntimeError:
        p_opt = (np.nan, np.nan, np.nan)
    fi_a, fi_b, fi_c = p_opt
    # rmse = np.sqrt(np.sum((firing_rates_model - fit_fun(amps_greater0, p_opt[0], p_opt[1], p_opt[2]))**2))

    FI_a = list(np.load(os.path.join(data_dir_FI_fit, 'FI_a.npy')))
    FI_b = list(np.load(os.path.join(data_dir_FI_fit, 'FI_b.npy')))
    FI_c = list(np.load(os.path.join(data_dir_FI_fit, 'FI_c.npy')))

    # plot in pdf
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    fig = plot_fi_curve(amps_greater0, firing_rates_model, os.path.join(save_dir_img, 'fi_curve'))
    plot_fi_curve_with_data(amps_greater0, firing_rates_model, firing_rates_data,
                                  os.path.join(save_dir_img, 'fi_curve'))
    pdf.savefig(fig)
    pl.close()

    np.save(os.path.join(save_dir_img, 'amps_greater0.npy'), amps_greater0)
    np.save(os.path.join(save_dir_img, 'firing_rates.npy'), firing_rates_model)

    fig = plot_IV_traces(amps[amps_greater0_idx], t_model, v_mat_model[amps_greater0_idx],
                         os.path.join(save_dir_img, 'traces'))
    pdf.savefig(fig)
    pl.close()

    jp1 = joint_plot_data_and_model(FI_a, FI_b, fi_a, fi_b, 'Scaling', 'Shift',
                                    os.path.join(save_dir_img, 'fi_curve', 'fit'))
    jp2 = joint_plot_data_and_model(FI_a, FI_c, fi_a, fi_c, 'Scaling', 'Exponent',
                                    os.path.join(save_dir_img, 'fi_curve', 'fit'))
    jp3 = joint_plot_data_and_model(FI_c, FI_b, fi_c, fi_b, 'Exponent', 'Shift',
                                    os.path.join(save_dir_img, 'fi_curve', 'fit'))
    pdf.savefig(jp1)
    pdf.savefig(jp2)
    pdf.savefig(jp3)
    pl.close()

    fig = plot_v(t_mat[0], v_mat_model[0], 'r', os.path.join(save_dir_img, 'sag'))
    pdf.savefig(fig)
    pl.close()

    fig = joint_plot_data_and_model(sag_amps_data, v_deflections_data, sag_amp, v_deflection, 'Sag Amplitude',
                                'Voltage Deflection', os.path.join(save_dir_img, 'sag'))
    pdf.savefig(fig)
    pl.close()


def use_amps_greater0(amps, amps_greater0_idx, firing_rates_data, firing_rates_model):
    amps_greater0 = amps[amps_greater0_idx]
    firing_rates_data = firing_rates_data[amps_greater0_idx]
    firing_rates_model = firing_rates_model[amps_greater0_idx]
    return amps_greater0, firing_rates_data, firing_rates_model


def sort_according_to_amps(amps, firing_rates_data, firing_rates_model, v_mat_data, v_mat_model):
    idx_sort = np.argsort(amps)
    amps = amps[idx_sort]
    firing_rates_data = firing_rates_data[idx_sort]
    firing_rates_model = firing_rates_model[idx_sort]
    v_mat_data = np.array(v_mat_data)[idx_sort]
    v_mat_model = np.array(v_mat_model)[idx_sort]
    return amps, firing_rates_data, firing_rates_model, v_mat_data, v_mat_model


def plot_IV_traces(amps, t_model, v_mat_model, save_dir_img=None):
    fig, ax = pl.subplots(20, 1, sharex=True, figsize=(21, 29.7))
    for i, (amp, v_trace_model) in enumerate(zip(amps[1:21], v_mat_model[1:21])):
        ax[i].plot(t_model, v_trace_model, 'r', label='$i_{amp}: $ %.2f' % amp)
        ax[i].set_ylim(-80, 60)
        ax[i].set_xlim(200, 850)
        ax[i].legend(fontsize=14)
    fig.text(0.06, 0.5, 'Membrane Potential (mV)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.06, 'Time (ms)', ha='center', fontsize=14)
    if save_dir_img is not None:
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)
        pl.savefig(os.path.join(save_dir_img, 'IV_subplots.pdf'))
    return fig


def plot_fi_curve(amps_greater0, firing_rates_model, save_dir_img=None):
    fig = pl.figure()
    pl.plot(amps_greater0, firing_rates_model, '-or', label='Model')
    pl.ylim([0, 100])
    pl.xlabel('Current (nA)')
    pl.ylabel('Firing rate (APs/s)')
    pl.tight_layout()
    if save_dir_img is not None:
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)
        pl.savefig(os.path.join(save_dir_img, 'fIcurve.png'))
    return fig


def plot_fi_curve_with_data(amps_greater0, firing_rates_model, firing_rates_data, save_dir_img=None):
    fig = pl.figure()
    pl.plot(amps_greater0, firing_rates_data, '-ok', label='Exp. Data')
    pl.plot(amps_greater0, firing_rates_model, '-or', label='Model')
    pl.ylim([0, 100])
    pl.xlabel('Current (nA)')
    pl.ylabel('Firing rate (APs/s)')
    pl.legend(loc='upper left')
    pl.tight_layout()
    if save_dir_img is not None:
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)
        pl.savefig(os.path.join(save_dir_img, 'fIcurve_with_data.png'))
    return fig


def get_slow_ramp(start_idx, end_idx, len_idx, step_amp):
    i_exp = np.zeros(len_idx)
    i_exp[start_idx:end_idx] = np.linspace(0, step_amp, end_idx - start_idx)
    return i_exp


def get_slow_ramp_reverse(start_idx, end_idx, len_idx, step_amp):
    i_exp = np.zeros(len_idx)
    i_exp[start_idx:end_idx] = np.linspace(1, step_amp, end_idx - start_idx)
    return i_exp


def get_step(start_idx, end_idx, len_idx, step_amp):
    i_exp = np.zeros(len_idx)
    i_exp[start_idx:end_idx] = step_amp
    return i_exp


def get_IV(cell, step_amp, step_fun, step_st_ms, step_end_ms, tstop, v_init=-75, dt=0.001):
    i_exp = step_fun(to_idx(step_st_ms, dt), to_idx(step_end_ms, dt), to_idx(tstop, dt)+1, step_amp)

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init, 'tstop': tstop,
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    return v, t, i_exp


def simulate_IV(cell, step_amp, v_init=-75):
    dt = 0.01
    tstop = 1150  # ms
    protocol = 'IV'
    sweep_idxs = [get_sweep_index_for_amp(step_amp, protocol)]
    i_inj = get_i_inj_from_function(protocol, sweep_idxs, tstop, dt)[0]

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': v_init, 'tstop': tstop,
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)
    return v, t


def compute_time_to_1st_spike(v_traces, i_traces, t_trace):

    start_step = np.nonzero(i_traces[0])[0][0]
    amps = np.array([i_inj[start_step] for i_inj in i_traces])

    time_to_1st_spike = np.zeros(len(amps))
    for i, amp in enumerate(amps):
        AP_onsets = get_AP_onset_idxs(v_traces[i], threshold=0)
        if len(AP_onsets) == 0:
            time_to_1st_spike[i] = np.nan
        else:
            time_to_1st_spike[i] = t_trace[AP_onsets[0]] - t_trace[start_step]
    return amps, time_to_1st_spike