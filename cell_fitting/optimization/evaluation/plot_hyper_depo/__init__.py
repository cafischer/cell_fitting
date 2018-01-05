import os
import numpy as np
import matplotlib.pyplot as pl
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_fitting.optimization.simulate import simulate_currents, iclamp_handling_onset
from cell_characteristics import to_idx
from cell_fitting.read_heka import get_i_inj_hyper_depo_ramp
from cell_fitting.util import init_nan
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_spike_characteristics
from cell_fitting.optimization.evaluation.plot_hyper_depo.plot_hyper_depo_summary import plot_linear_fit_results, \
    plot_slope_rmse


def evaluate_hyper_depo(pdf, cell, data_dir_slopes, save_dir):
    save_dir_img = os.path.join(save_dir, 'img', 'hyper_depo')
    step_amps = np.array([-0.25, -0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.25])
    ramp_amp = 5.0  # nA
    step_start = 200
    ramp_start = 600
    dt = 0.01
    tstop = 1000  # ms

    # simulate / load
    spike_characteristic_params = get_spike_characteristics_dict()
    return_characteristics = ['DAP_amp', 'DAP_deflection', 'DAP_width', 'fAHP_amp', 'DAP_time']

    t_mat, v_mat, i_inj_mat, _ = simulate_hyper_depo(cell, dt, ramp_amp, step_amps, tstop, v_init=-75, celsius=35,
                                                     onset=200)

    # evaluate
    spike_characteristics_mat, v_step = get_spike_characteristics_and_vstep(v_mat, t_mat, spike_characteristic_params,
                                                                            return_characteristics, ramp_start,
                                                                            step_start)

    slopes_model, intercepts_model, rmses_model = plot_linear_fit_results([step_amps],
                                                                          'Step Current Amplitude (nA)',
                                                                          [spike_characteristics_mat],
                                                                          return_characteristics,
                                                                          [''], None)
    spike_characteristic_mat_per_cell = np.load(os.path.join(data_dir_slopes,
                                                             'spike_characteristic_mat_per_cell.npy'))
    amps_per_cell = np.load(os.path.join(data_dir_slopes, 'amps_per_cell.npy'))
    cell_ids = np.load(os.path.join(data_dir_slopes, 'cell_ids.npy'))
    slopes_data, intercepts_data, rmses_data = plot_linear_fit_results(amps_per_cell, 'Step Current Amplitude (nA)',
                                                                       spike_characteristic_mat_per_cell,
                                                                       return_characteristics,
                                                                       cell_ids, None)

    # plot in pdf
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    fig = plot_hyper_depo(step_amps, t_mat, v_mat, save_dir_img)
    pdf.savefig(fig)
    pl.close()

    figs = plot_slope_rmse(return_characteristics, 'Step Current Amplitude (nA)', rmses_data, slopes_data,
                    slopes_model, rmses_model, [''], save_dir_img)
    for fig in figs:
        pdf.savefig(fig)
    pl.close()


def simulate_hyper_depo(cell, dt, ramp_amp, step_amps, tstop, v_init=-75, celsius=35, onset=200):
    len_t = to_idx(tstop, dt) + 1
    currents = np.zeros((len(step_amps)), dtype=object)
    v_mat = np.zeros((len(step_amps), len_t))
    t_mat = np.zeros((len(step_amps), len_t))
    i_inj_mat = np.zeros((len(step_amps), len_t))
    for i, step_amp in enumerate(step_amps):
        i_inj_mat[i, :] = get_i_inj_hyper_depo_ramp(step_amp=step_amp, ramp_amp=ramp_amp, tstop=tstop, dt=dt)
        simulation_params = {'sec': ('soma', None), 'i_inj': i_inj_mat[i], 'v_init': v_init, 'tstop': tstop,
                             'dt': dt, 'celsius': celsius, 'onset': onset}
        v_mat[i, :], t_mat[i, :], _ = iclamp_handling_onset(cell, **simulation_params)
        currents[i], channel_list = simulate_currents(cell, simulation_params, plot=False)
    return t_mat, v_mat, i_inj_mat, currents


def get_spike_characteristics_and_vstep(v_traces, t_traces, spike_characteristic_params, return_characteristics,
                                        ramp_start, step_start):
    spike_characteristics_mat = init_nan((len(v_traces), len(return_characteristics)))
    v_step = init_nan(len(v_traces))
    for i, (v, t) in enumerate(zip(v_traces, t_traces)):
        onset_idxs_after_ramp = get_AP_onset_idxs(v[to_idx(ramp_start, t[1] - t[0]):to_idx(ramp_start + 10, t[1] - t[0])],
                                                 spike_characteristic_params['AP_threshold'])
        onset_idxs_all = get_AP_onset_idxs(v, spike_characteristic_params['AP_threshold'])

        if len(onset_idxs_after_ramp) >= 1 and len(onset_idxs_all) - len(onset_idxs_after_ramp) == 0:
            v_step[i] = np.mean(v[to_idx(step_start + (ramp_start-step_start)/2, t[1] - t[0]): to_idx(ramp_start, t[1] - t[0])])
            v_rest = np.mean(v[0:to_idx(step_start, t[1] - t[0])])
            std_idx_times = (0, 10)  # rather short so that no global changes interfere
            spike_characteristics_mat[i, :] = get_spike_characteristics(np.array(v, dtype=float),
                                                                        np.array(t, dtype=float),
                                                                        return_characteristics,
                                                                        v_rest=v_rest, std_idx_times=std_idx_times,
                                                                        check=False,
                                                                        **spike_characteristic_params)
            # set to nan if spike on DAP
            if spike_characteristics_mat[i, np.array(return_characteristics) == 'DAP_amp'] > 50:
                spike_characteristics_mat[i, :] = init_nan(len(return_characteristics))
    return spike_characteristics_mat, v_step


def plot_hyper_depo(step_amps, t_mat, v_mat, save_dir_img=None):
    c_map = pl.cm.get_cmap('plasma')
    colors = c_map(np.linspace(0, 1, len(step_amps)))

    fig = pl.figure()
    for j, step_amp in enumerate(step_amps):
        pl.plot(t_mat[j], v_mat[j], c=colors[j], label='%.2f (nA)' % step_amp)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.legend(fontsize=10)
    pl.xlim(595, 645)
    pl.ylim(-90, -40)
    pl.tight_layout()
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'v_zoom.png'))
    return fig