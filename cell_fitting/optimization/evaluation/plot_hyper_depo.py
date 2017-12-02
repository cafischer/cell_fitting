import os

import numpy as np
import pylab as pl
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_spike_characteristics
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import simulate_currents, iclamp_handling_onset
from cell_fitting.util import init_nan
from cell_fitting.read_heka import get_i_inj_hyper_depo_ramp
from cell_fitting.read_heka import get_v_and_t_from_heka

pl.style.use('paper')

__author__ = 'caro'


def simulate_hyper_depo_ramp(cell, data_dir):
    step_amps = np.arange(-0.2, 0.21, 0.05)
    ramp_amp = 5  # nA
    dt = 0.01
    tstop = 1000  # ms

    AP_threshold = -30  # mV
    AP_interval = 2.5  # ms
    fAHP_interval = 4.0  # ms
    AP_width_before_onset = 2  # ms
    DAP_interval = 10  # ms
    order_fAHP_min = 1.0  # ms (how many points to consider for the minimum)
    order_DAP_max = 1.0  # ms (how many points to consider for the minimum)
    min_dist_to_DAP_max = 0.5  # ms
    k_splines = 3
    s_splines = 0

    t_exp = np.arange(0, tstop + dt, dt)
    v = np.zeros([len(step_amps), len(t_exp)])

    currents = []
    DAP_amps = init_nan(len(step_amps))
    DAP_deflections = init_nan(len(step_amps))

    for j, step_amp in enumerate(step_amps):
        i_exp = get_i_inj_hyper_depo_ramp(step_amp=step_amp, ramp_amp=ramp_amp, tstop=tstop, dt=dt)

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
                                                            AP_interval, AP_width_before_onset, fAHP_interval,
                                                            std_idx_times, k_splines, s_splines, order_fAHP_min,
                                                            DAP_interval, order_DAP_max, min_dist_to_DAP_max,
                                                            check=False)
        DAP_amps[j] = DAP_amp
        DAP_deflections[j] = DAP_deflection

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'hyperdap')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    c_map = pl.cm.get_cmap('plasma')
    colors = c_map(np.linspace(0, 1, len(step_amps)))

    pl.figure(figsize=(8, 6))
    for j, step_amp in enumerate(step_amps):
        pl.plot(t, v[j], c=colors[j], label='%.2f (nA)' % step_amp)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend(loc='upper right')
    pl.tight_layout()
    pl.xlim(0, t[-1])
    pl.savefig(os.path.join(save_dir_img, 'hyperDAP.png'))
    #pl.show()

    pl.figure(figsize=(8, 6))
    for j, step_amp in enumerate(step_amps):
        pl.plot(t, v[j], c=colors[j], label='%.2f (nA)' % step_amp)
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
    pl.plot(np.array(step_amps)[not_nan], np.array(DAP_amps)[not_nan], 'ok')
    pl.xlabel('Current Amplitude (nA)')
    pl.ylabel('DAP Amplitude (mV)')
    pl.xticks(step_amps)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_amp.png'))
    # pl.show()

    not_nan = ~np.isnan(DAP_deflections)
    pl.figure()
    pl.plot(np.array(step_amps)[not_nan], np.array(DAP_deflections)[not_nan], 'ok')
    pl.xlabel('Current Amplitude (nA)')
    pl.ylabel('DAP Deflection (mV)')
    pl.xticks(step_amps)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_deflection.png'))
    #pl.show()

    # plot currents
    pl.figure()
    colors = c_map(np.linspace(0, 1, len(currents[0])))
    for j, step_amp in enumerate(step_amps):
        for k, current in enumerate(currents[j]):
            pl.plot(t, -1*current, c=colors[k], label=channel_list[k])
    pl.xlabel('Time (ms)')
    pl.ylabel('Current (mA/cm$^2$)')
    pl.xlim(595, 645)
    pl.tight_layout()
    #pl.show()

    # compare with data
    for p_idx in [0, 1, 2, 3]:
        if p_idx == 0:
            v_data, t_data = get_v_and_t_from_heka(data_dir, 'hyperRampTester')
        else:
            v_data_tmp, t_data_tmp = get_v_and_t_from_heka(data_dir, 'hyperRampTester'+'('+str(p_idx)+')')
            v_data = np.concatenate((v_data, v_data_tmp), axis=0)
            t_data = np.concatenate((t_data, t_data_tmp), axis=0)
    for p_idx in [0, 1, 2, 3]:
        if p_idx == 0:
            v_data_tmp, t_data_tmp = get_v_and_t_from_heka(data_dir, 'depoRampTester')
        else:
            v_data_tmp, t_data_tmp = get_v_and_t_from_heka(data_dir, 'depoRampTester'+'('+str(p_idx)+')')
        v_data = np.concatenate((v_data, v_data_tmp), axis=0)
        t_data = np.concatenate((t_data, t_data_tmp), axis=0)

    c_map = pl.cm.get_cmap('plasma')
    colors = c_map(np.linspace(0, 1, len(step_amps)))
    pl.figure(figsize=(8, 6))
    for j, step_amp in enumerate(step_amps):
        pl.plot(t, v[j], c=colors[j], label='%.2f (nA)' % step_amp)
    for k in range(len(v_data)):
        pl.plot(t_data[k], v_data[k]-8, 'k')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.xlim(595, 645)
    pl.ylim(-95, -40)
    pl.tight_layout()
    pl.show()

if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2013_12_11a.dat'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # simulate
    simulate_hyper_depo_ramp(cell, data_dir)