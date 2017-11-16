import os

import numpy as np
import pylab as pl
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_spike_characteristics
from nrn_wrapper import Cell

from cell_fitting.optimization.simulate import simulate_currents, iclamp_handling_onset
from cell_fitting.util import init_nan

pl.style.use('paper')

__author__ = 'caro'


def get_ramp(start_idx, end_idx, amp_before, ramp_amp, amp_after):
    diff_idx = end_idx - start_idx
    half_diff_up = int(round(diff_idx / 2)) - 1
    half_diff_down = int(round(diff_idx / 2)) + 1  # peak is one earlier
    if diff_idx % 2 != 0:
        half_diff_down += 1
    i_exp = np.zeros(diff_idx)
    i_exp[:half_diff_up] = np.linspace(amp_before, ramp_amp, half_diff_up)
    i_exp[half_diff_up:] = np.linspace(ramp_amp, amp_after, half_diff_down+1)[1:]
    return i_exp


def hyperpolarize_ramp(cell):
    """
    params:
    hyperamps = np.arange(-0.25, 0.26, 0.05)  # nA
    ramp_amp = 8  # nA
    dt = 0.01
    hyp_st_ms = 200  # ms
    hyp_end_ms = 600  # ms
    ramp_end_ms = 602  # ms
    tstop = 1000  # ms
    """

    #hyperamps = np.arange(-0.25, 0.26, 0.05)  # nA  #TODO
    hyperamps = np.arange(-0.2, 0.21, 0.05)
    ramp_amp = 8  # nA
    dt = 0.01
    hyp_st_ms = 200  # ms
    hyp_end_ms = 600  # ms
    ramp_end_ms = 602  # ms
    tstop = 1000  # ms

    AP_threshold = -30  # mV
    AP_interval = 4.0 # TODO  # ms (also used as interval for fAHP)
    AP_width_before_onset = 2  # ms
    DAP_interval = 10  # ms
    order_fAHP_min = 1.0  # ms (how many points to consider for the minimum)
    order_DAP_max = 1.0  # ms (how many points to consider for the minimum)
    min_dist_to_DAP_max = 0.5  # ms
    k_splines = 3
    s_splines = 0

    hyp_st = int(round(hyp_st_ms / dt))
    hyp_end = int(round(hyp_end_ms / dt))
    ramp_end = int(round(ramp_end_ms / dt)) + 1

    t_exp = np.arange(0, tstop + dt, dt)

    v = np.zeros([len(hyperamps), len(t_exp)])
    currents = []
    DAP_amps = init_nan(len(hyperamps))
    DAP_deflections = init_nan(len(hyperamps))
    for j, hyper_amp in enumerate(hyperamps):
        i_exp = np.zeros(len(t_exp))
        i_exp[hyp_st:hyp_end] = hyper_amp
        i_exp[hyp_end:ramp_end] = get_ramp(hyp_end, ramp_end, hyper_amp, ramp_amp, 0)

        # get simulation parameters
        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -59, 'tstop': t_exp[-1],
                             'dt': dt, 'celsius': 35, 'onset': 200}

        # record v
        v[j], t, _ = iclamp_handling_onset(cell, **simulation_params)

        currents_tmp, channel_list = simulate_currents(cell, simulation_params, plot=False)
        currents.append(currents_tmp)

        # get DAP amp and deflection
        v_rest = np.mean(v[j][0:to_idx(100, t[1] - t[0])])
        std_idx_times = (0, 100)
        DAP_amp, DAP_deflection = get_spike_characteristics(v[j][to_idx(600, t[1] - t[0]):],
                                                            t[to_idx(600, t[1] - t[0]):],
                                                            ['DAP_amp', 'DAP_deflection'],
                                                            v_rest, AP_threshold,
                                                            AP_interval, AP_width_before_onset, std_idx_times,
                                                            k_splines, s_splines, order_fAHP_min, DAP_interval,
                                                            order_DAP_max, min_dist_to_DAP_max, check=False)
        DAP_amps[j] = DAP_amp
        DAP_deflections[j] = DAP_deflection

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'hyperdap')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    c_map = pl.cm.get_cmap('plasma')
    colors = c_map(np.linspace(0, 1, len(hyperamps)))

    pl.figure(figsize=(8, 6))
    for j, hyper_amp in enumerate(hyperamps):
        pl.plot(t, v[j], c=colors[j], label=str(np.round(hyper_amp, 2)) + ' nA')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend(loc='upper right')
    pl.tight_layout()
    pl.xlim(0, t[-1])
    pl.savefig(os.path.join(save_dir_img, 'hyperDAP.png'))
    #pl.show()

    pl.figure(figsize=(8, 6))
    for j, hyper_amp in enumerate(hyperamps):
        pl.plot(t, v[j], c=colors[j], label=str(np.round(hyper_amp, 2)) + ' nA')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.xlim(595, 645)
    pl.ylim(-95, -40)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'hyperDAP_zoom.png'))
    #pl.show()

    DAP_amps[DAP_amps > 50] = np.nan  # take out spikes on DAP
    DAP_deflections[DAP_amps > 50] = np.nan

    not_nan = ~np.isnan(DAP_amps)
    pl.figure()
    pl.plot(np.array(hyperamps)[not_nan], np.array(DAP_amps)[not_nan], 'ok')
    pl.xlabel('Current Amplitude (nA)')
    pl.ylabel('DAP Amplitude (mV)')
    pl.xticks(hyperamps)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_amp.png'))
    # pl.show()

    not_nan = ~np.isnan(DAP_deflections)
    pl.figure()
    pl.plot(np.array(hyperamps)[not_nan], np.array(DAP_deflections)[not_nan], 'ok')
    pl.xlabel('Current Amplitude (nA)')
    pl.ylabel('DAP Deflection (mV)')
    pl.xticks(hyperamps)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_deflection.png'))
    #pl.show()

    # plot currents
    pl.figure()
    colors = c_map(np.linspace(0, 1, len(currents[0])))
    for j, hyper_amp in enumerate(hyperamps):
        for k, current in enumerate(currents[j]):
            pl.plot(t, -1*current, c=colors[k], label=channel_list[k])
    pl.xlabel('Time (ms)')
    pl.ylabel('Current (mA/cm$^2$)')
    pl.xlim(595, 645)
    pl.tight_layout()
    #pl.show()



if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # start hyperdap
    hyperpolarize_ramp(cell)