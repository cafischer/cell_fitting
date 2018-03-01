from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.investigate_grid_cell_stimuli.model_noise.with_OU import ou_noise_input
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx
from cell_characteristics import to_idx
from neuron import h
from sklearn.decomposition import FastICA
from itertools import combinations


if __name__ == '__main__':
    # parameters
    model_ids = range(1, 7)
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '../../../model/channels/vavoulis'
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")
    load_mechanism_dir(mechanism_dir)

    noise_params = {'g_e0': 0.01, 'g_i0': 0.05, 'std_e': 0.004, 'std_i': 0.004, 'tau_e': 3.0, 'tau_i': 10.0,
                    'E_e': 0, 'E_i': -75}

    before_AP = 0
    after_AP = 25
    cut_before_AP = before_AP + 1.25
    cut_after_AP = after_AP - 25

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
        g_e = h.Vector()
        g_e.record(ou_process._ref_g_e)
        g_i = h.Vector()
        g_i.record(ou_process._ref_g_i)
        simulation_params = {'sec': ('soma', None), 'i_inj': np.zeros(to_idx(tstop, dt)), 'v_init': v_init,
                             'tstop': tstop, 'dt': dt, 'celsius': celsius, 'onset': onset}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)
        i_noise = -1 * np.array(i_noise)[to_idx(onset, dt):]  # -1: follows convention of ionic currents, -1 makes pos. current depolarizing
        g_e = np.array(g_e)[to_idx(onset, dt):]
        g_i = np.array(g_i)[to_idx(onset, dt):]

        # find all spikes
        AP_threshold = np.min(v) + 2. / 3 * np.abs(np.min(v) - np.max(v)) - 5
        onset_idxs = get_AP_onset_idxs(v, AP_threshold)
        
        if len(onset_idxs) > 0:
            onset_idxs = np.insert(onset_idxs, len(onset_idxs), len(v))
            AP_max_idxs = [get_AP_max_idx(v, onset_idx, onset_next_idx) for (onset_idx, onset_next_idx)
                           in zip(onset_idxs[:-1], onset_idxs[1:])]
            # pl.figure()
            # pl.plot(t, v)
            # pl.plot(t[AP_max_idxs], v[AP_max_idxs], 'or')
            # pl.show()

            # take window around each spike
            before_AP_idx = to_idx(before_AP, dt)
            after_AP_idx = to_idx(after_AP, dt)
            cut_before_AP_idx = to_idx(cut_before_AP, dt)
            cut_after_AP_idx = to_idx(cut_after_AP, dt)
            v_APs = []
            i_APs = []
            g_e_APs = []
            g_i_APs = []
            for AP_max_idx in AP_max_idxs:
                if (AP_max_idx is not None
                        and AP_max_idx - before_AP_idx >= 0
                        and AP_max_idx + after_AP_idx + 1 <= len(v)):  # able to draw window
                    v_AP = v[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1]
                    if len(get_AP_onset_idxs(v_AP, AP_threshold)) == 0:  # no bursts (1st AP should not be detected as it starts from the max)
                        v_APs.append(v_AP)
                        i_APs.append(i_noise[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1])
                        g_e_APs.append(g_e[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1])
                        g_i_APs.append(g_i[onset_idx - before_AP_idx:onset_idx + after_AP_idx + 1])
        v_APs = np.vstack(v_APs)
        i_APs = np.vstack(i_APs)
        g_e_APs = np.vstack(g_e_APs)
        g_i_APs = np.vstack(g_i_APs)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt

        # STA on v
        spike_triggered_avg_v = np.mean(v_APs, 0)
        spike_triggered_std_v = np.std(v_APs, 0)

        # STA on i
        spike_triggered_avg_i = np.mean(i_APs, 0)
        spike_triggered_std_i = np.std(i_APs, 0)
        spike_triggered_avg_g_e = np.mean(g_e_APs, 0)
        spike_triggered_std_g_e = np.std(g_e_APs, 0)
        spike_triggered_avg_g_i = np.mean(g_i_APs, 0)
        spike_triggered_std_g_i = np.std(g_i_APs, 0)

        # save and plot
        save_dir_img = os.path.join(save_dir, str(model_id), 'img', 'STA', 'without_pulse')
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        print '#APs: ' + str(len(v_APs))

        pl.figure()
        pl.plot(t, v, 'k')
        #pl.plot(t, i_noise, 'b')
        pl.ylabel('Membrane potential (mV)', fontsize=16)
        pl.xlabel('Time (ms)', fontsize=16)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'v.png'))

        pl.figure()
        for v_AP in v_APs:
            pl.plot(t_AP, v_AP)
        pl.ylabel('Membrane potential (mV)', fontsize=16)
        pl.xlabel('Time (ms)', fontsize=16)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'v_APs.png'))

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
        #pl.show()

        pl.figure()
        pl.plot(t_AP, spike_triggered_avg_g_e, 'r')
        pl.fill_between(t_AP, spike_triggered_avg_g_e + spike_triggered_std_g_e,
                        spike_triggered_avg_g_e - spike_triggered_std_g_e,
                        facecolor='r', alpha=0.5)
        pl.xlabel('Time (ms)')
        pl.ylabel('Excitatory Conductance (uS)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'STA_g_e.png'))
        #pl.show()

        pl.figure()
        pl.plot(t_AP, spike_triggered_avg_g_i, 'r')
        pl.fill_between(t_AP, spike_triggered_avg_g_i + spike_triggered_std_g_i,
                        spike_triggered_avg_g_i - spike_triggered_std_g_i,
                        facecolor='r', alpha=0.5)
        pl.xlabel('Time (ms)')
        pl.ylabel('Inhibitory Conductance (uS)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'STA_g_i.png'))
        #pl.show()

        # STC
        if len(v_APs) > 10:
            len_v_AP = len(v_APs[0])
            v_APs = v_APs[:, cut_before_AP_idx:len_v_AP - cut_after_AP_idx]  # select smaller window around APs
            t_AP = t_AP[cut_before_AP_idx:len_v_AP - cut_after_AP_idx]
            v_APs = v_APs[:, ::10]  # downsample
            t_AP = t_AP[::10]
            v_APs_centered = v_APs - np.mean(v_APs, 0) #/ np.std(v_APs, 0) # TODO: standardizing
            cov = np.cov(v_APs.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            assert np.all([np.round(np.dot(v1, v2), 10) == 0 for v1, v2 in combinations(eigvecs.T, 2)])  # check orthogonality of eigvecs
            idx_sort = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx_sort]
            eigvecs = eigvecs[:, idx_sort]
            min_expl_var = 0.8
            n = np.where(np.cumsum(eigvals) / np.sum(eigvals) >= min_expl_var)[0][0]
            chosen_eigvecs = eigvecs[:, :n+1]
            back_transform = np.dot(v_APs_centered, np.dot(chosen_eigvecs, chosen_eigvecs.T)) + np.mean(v_APs, 0)
            expl_var = eigvals / np.sum(eigvals) * 100

            pl.figure()
            for vec in back_transform:
                pl.plot(t_AP, vec)
            pl.title('Backprojected AP Traces', fontsize=18)
            pl.ylabel('Membrane Potential (mV)')
            pl.xlabel('Time (ms)')
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'STC_backprojected_APs.png'))
            #pl.show()

            pl.figure()
            for vec in v_APs:
                pl.plot(t_AP, vec)
            pl.title('AP Traces', fontsize=18)
            pl.ylabel('Membrane Potential (mV)')
            pl.xlabel('Time (ms)')
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'STC_APs.png'))
            #pl.show()

            pl.figure()
            for i, vec in enumerate(chosen_eigvecs.T):
                pl.plot(t_AP, vec, label='expl. var.: %i %%' % int(round(expl_var[i])))
            pl.title('Eigenvectors', fontsize=18)
            pl.xlabel('Time (ms)')
            pl.tight_layout()
            pl.legend(loc='upper left', fontsize=10)
            pl.savefig(os.path.join(save_dir_img, 'STC_largest_eigenvecs.png'))
            #pl.show()

            pl.figure()
            pl.plot(np.arange(len(expl_var)), np.cumsum(expl_var), 'ok')
            pl.axhline(min_expl_var*100, 0, 1, color='0.5', linestyle='--',
                       label='%i %% expl. var.' % int(round(min_expl_var*100)))
            pl.title('Cumulative Explained Variance', fontsize=18)
            pl.ylabel('Percent')
            pl.xlabel('#')
            pl.ylim(0, 105)
            #pl.legend(fontsize=16)
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'STC_eigenvals.png'))
            #pl.show()

            # ICA
            ica = FastICA(n_components=3, whiten=True)
            ica_components = ica.fit_transform(v_APs.T)  # Reconstruct signals
            pl.figure()
            for vec in ica_components.T:
                pl.plot(t_AP, vec)
            pl.title('ICA Components', fontsize=18)
            pl.ylabel('Membrane Potential (mV)')
            pl.xlabel('Time (ms)')
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'ICA_components.png'))
            #pl.show()

        pl.close('all')