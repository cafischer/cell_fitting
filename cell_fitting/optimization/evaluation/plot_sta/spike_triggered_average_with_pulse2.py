from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.investigate_grid_cell_stimuli.model_noise.with_OU import ou_noise_input
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_characteristics import to_idx
from neuron import h
from cell_fitting.read_heka.i_inj_functions import get_i_inj_rampIV


if __name__ == '__main__':
    # parameters
    model_ids = range(1, 7)
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '../../../model/channels/vavoulis'
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")
    load_mechanism_dir(mechanism_dir)

    gi0s = np.arange(0, 0.21, 0.025)

    tstop = 200
    dt = 0.01
    v_init = -75
    celsius = 35
    onset = 200

    n_trials = 5
    AP_threshold = -20

    i_amp = 5.0  # nA
    ramp_start = 50.0  # ms
    ramp_peak = 50.8  # ms
    ramp_end = 52.0  # ms
    ramp_start_idx = to_idx(ramp_start, dt)

    before_AP = 50
    after_AP = 100
    before_AP_idx = to_idx(before_AP, dt)
    after_AP_idx = to_idx(after_AP, dt)

    #noise_params = {'g_e0': 0.003, 'g_i0': 0.05, 'std_e': 0.007, 'std_i': 0.006, 'tau_e': 2.4, 'tau_i': 5.0}
    noise_params = {'g_e0': 0.0, 'g_i0': 0.0, 'std_e': 0.0, 'std_i': 0.0, 'tau_e': 1.0, 'tau_i': 1.0}
    #noise_params['g_i0'] = 0.05

    seed = 1

    for model_id in model_ids:
        print model_id
        spike_triggered_avg_vs = []
        for gi0 in gi0s:
            noise_params['g_i0'] = gi0  # TODO g_i0
            v_APs = []
            for trial in range(n_trials):
                cell = Cell.from_modeldir(os.path.join(save_dir, str(model_id), 'cell.json'))

                ou_process = ou_noise_input(cell, **noise_params)
                ou_process.new_seed(trial + seed)

                # simulate
                i_noise = h.Vector()
                i_noise.record(ou_process._ref_i)
                g_e = h.Vector()
                g_e.record(ou_process._ref_g_e)
                g_i = h.Vector()
                g_i.record(ou_process._ref_g_i)

                i_inj = get_i_inj_rampIV(ramp_start, ramp_peak, ramp_end, 0, i_amp, 0, tstop, dt)

                simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': v_init,
                                     'tstop': tstop, 'dt': dt, 'celsius': celsius, 'onset': onset}
                v, t, _ = iclamp_handling_onset(cell, **simulation_params)
                i_noise = -1 * np.array(i_noise)[to_idx(onset, dt):]  # -1: follows convention of ionic currents, -1 makes pos. current depolarizing
                g_e = np.array(g_e)[to_idx(onset, dt):]
                g_i = np.array(g_i)[to_idx(onset, dt):]

                # take window around stimulus onset
                v_AP = v[ramp_start_idx - before_AP_idx:ramp_start_idx + after_AP_idx + 1]
                onset_idxs = get_AP_onset_idxs(v_AP, AP_threshold)
                if len(onset_idxs) == 1 and ramp_start_idx < onset_idxs[0] < ramp_start_idx + to_idx(2, dt):  # no bursts and AP at stimulus onset
                    i_AP = i_noise[ramp_start_idx - before_AP_idx:ramp_start_idx + after_AP_idx + 1]
                    v_APs.append(v_AP)

            if len(v_APs) > 0:
                v_APs = np.vstack(v_APs)
                t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt

                # STA on v
                spike_triggered_avg_v = np.mean(v_APs, 0)
                spike_triggered_std_v = np.std(v_APs, 0)

                spike_triggered_avg_vs.append(spike_triggered_avg_v)

        # save and plot
        save_dir_img = os.path.join(save_dir, str(model_id), 'img', 'STA', 'with_pulse')
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        pl.figure()
        colors = pl.cm.plasma(np.linspace(0, 1, len(gi0s)))
        for i, spike_triggered_avg_v in enumerate(spike_triggered_avg_vs):
            pl.plot(t_AP, spike_triggered_avg_v, label='%.2f'%gi0s[i], color=colors[i])
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'gi0_changing_more_ge0_0.png'))
        #pl.show()