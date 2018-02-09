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


if __name__ == '__main__':
    # parameters
    model_ids = range(1, 7)
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '../../model/channels/vavoulis'
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")
    load_mechanism_dir(mechanism_dir)

    noise_params = {'g_e0': 0.003, 'g_i0': 0.05, 'std_e': 0.007, 'std_i': 0.006, 'tau_e': 2.4, 'tau_i': 5.0}

    before_AP = 50
    after_AP = 50

    tstop = 50000
    dt = 0.01
    v_init = -75
    celsius = 35
    onset = 200

    seed = 1


    for model_id in model_ids:
        # create cell
        cell = Cell.from_modeldir(os.path.join(save_dir, str(model_id), 'cell.json'))

        ou_process = ou_noise_input(cell, **noise_params)
        ou_process.new_seed(seed)

        # simulate
        i_noise = h.Vector()
        i_noise.record(ou_process._ref_i)
        simulation_params = {'sec': ('soma', None), 'i_inj': np.zeros(to_idx(tstop, dt)), 'v_init': v_init,
                             'tstop': tstop, 'dt': dt, 'celsius': celsius, 'onset': onset}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)
        i_noise = -1 * np.array(i_noise)[to_idx(onset, dt):]  # -1: follows convention of ionic currents, -1 makes pos. current depolarizing

        # find all spikes
        AP_threshold = np.min(v) + 2. / 3 * np.abs(np.min(v) - np.max(v)) - 5
        onset_idxs = get_AP_onset_idxs(v, AP_threshold)

        # take window around each spike
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)
        v_APs = []
        i_APs = []
        for onset_idx in onset_idxs:
            if onset_idx - before_AP_idx >= 0 and onset_idx + after_AP_idx + 1 <= len(v):  # able to draw window
                v_AP = v[onset_idx - before_AP_idx:onset_idx + after_AP_idx + 1]
                if len(get_AP_onset_idxs(v_AP, AP_threshold)) == 1:  # no bursts
                    v_APs.append(v_AP)
                    i_APs.append(i_noise[onset_idx - before_AP_idx:onset_idx + after_AP_idx + 1])
        v_APs = np.vstack(v_APs)
        i_APs = np.vstack(i_APs)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt

        # STA on v
        spike_triggered_avg_v = np.mean(v_APs, 0)
        spike_triggered_std_v = np.std(v_APs, 0)

        # STA on i
        spike_triggered_avg_i = np.mean(i_APs, 0)
        spike_triggered_std_i = np.std(i_APs, 0)

        # save and plot
        save_dir_img = os.path.join(save_dir, str(model_id), 'img', 'STA')
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        print '#APs: ' + str(len(v_APs))

        pl.figure()
        pl.plot(t, v, 'k')
        #pl.plot(t, i_noise, 'b')
        pl.ylabel('Membrane potential (mV)', fontsize=16)
        pl.xlabel('Time (ms)', fontsize=16)
        pl.savefig(os.path.join(save_dir_img, 'v.png'))

        pl.figure()
        for v_AP in v_APs:
            pl.plot(t_AP, v_AP)

        pl.figure()
        pl.plot(t_AP, spike_triggered_avg_v, 'r')
        pl.fill_between(t_AP, spike_triggered_avg_v + spike_triggered_std_v, spike_triggered_avg_v - spike_triggered_std_v,
                        facecolor='r', alpha=0.5)
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'STA_v.png'))
        #pl.show()

        pl.figure()
        pl.plot(t_AP, spike_triggered_avg_i, 'r')
        pl.fill_between(t_AP, spike_triggered_avg_i + spike_triggered_std_i,
                        spike_triggered_avg_i - spike_triggered_std_i,
                        facecolor='r', alpha=0.5)
        pl.xlabel('Time (ms)')
        pl.ylabel('Synaptic Current (nA)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'STA_i_inj.png'))
        pl.show()