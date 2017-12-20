import os
import numpy as np
import pylab as pl
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.simulate import simulate_currents, iclamp_handling_onset
from cell_fitting.read_heka import get_i_inj_hyper_depo_ramp
from cell_fitting.data.plot_hyper_depo.plot_hyper_depo import get_spike_characteristics_and_vstep
pl.style.use('paper')

__author__ = 'caro'

characteristic_name_dict = {'DAP_amp': 'DAP Amplitude', 'DAP_deflection': 'DAP Deflection', 'DAP_width': 'DAP Width',
                            'DAP_time': 'DAP Time', 'fAHP_amp': 'fAHP Amplitude'}
characteristic_unit_dict = {'DAP_amp': 'mV', 'DAP_deflection': 'mV', 'DAP_width': 'ms',
                            'DAP_time': 'ms', 'fAHP_amp': 'mV'}


def simulate_hyper_depo_ramp(cell, save_dir):
    step_amps = np.array([-0.25, -0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.25])
    ramp_amp = 5  # nA
    step_start = 200
    ramp_start = 600
    dt = 0.01
    tstop = 1000  # ms

    spike_characteristic_params = {'AP_threshold': 0, 'AP_interval': 2.5, 'AP_width_before_onset': 2,
                                   'fAHP_interval': 4.0, 'DAP_interval': 10, 'order_fAHP_min': 1.0,
                                   'order_DAP_max': 1.0, 'min_dist_to_DAP_max': 0.5, 'k_splines': 3, 's_splines': 0}
    return_characteristics = ['DAP_amp', 'DAP_deflection', 'DAP_width', 'fAHP_amp', 'DAP_time']

    t_traces, v_traces, currents = get_v_traces_t_traces_and_currents(cell, dt, ramp_amp, step_amps, tstop)

    spike_characteristics_mat, v_step = get_spike_characteristics_and_vstep(v_traces, t_traces,
                                                                            spike_characteristic_params,
                                                                            return_characteristics, ramp_start, step_start)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'hyper_depo')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'spike_characteristics_mat.npy'), spike_characteristics_mat)
    np.save(os.path.join(save_dir_img, 'amps.npy'), step_amps)
    np.save(os.path.join(save_dir_img, 'v_step.npy'), v_step)

    c_map = pl.cm.get_cmap('plasma')
    colors = c_map(np.linspace(0, 1, len(step_amps)))

    pl.figure()
    for j, step_amp in enumerate(step_amps):
        pl.plot(t_traces[j], v_traces[j], c=colors[j], label='%.2f (nA)' % step_amp)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.legend(fontsize=10)
    pl.xlim(100, 800)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'v.png'))
    #pl.show()

    pl.figure()
    for j, step_amp in enumerate(step_amps):
        pl.plot(t_traces[j], v_traces[j], c=colors[j], label='%.2f (nA)' % step_amp)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend(fontsize=10)
    pl.xlim(595, 645)
    pl.ylim(-95, -40)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'v_zoom.png'))
    #pl.show()

    # pl.figure()
    # for j, step_amp in enumerate(step_amps):
    #     pl.plot(t, i_inj[j], c=colors[j], label='%.2f (nA)' % step_amp)
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Current (nA)')
    # pl.legend(loc='upper right')
    # pl.xlim(100, 800)
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'i_inj.png'))
    # #pl.show()

    for i, spike_characteristic in enumerate(spike_characteristics_mat.T):
        not_nan = ~np.isnan(spike_characteristic)
        pl.figure()
        pl.plot(np.array(step_amps)[not_nan], np.array(spike_characteristic)[not_nan], 'ok')
        pl.xlabel('Step Current Amplitude (nA)')
        pl.ylabel('$RMSE_{%s}$ (%s)' % (characteristic_name_dict[return_characteristics[i]].replace(' ', '\ '),
                                               characteristic_unit_dict[return_characteristics[i]]))
        pl.xlim(min(step_amps)-0.05, max(step_amps)+0.05)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, return_characteristics[i]+'.png'))
        #pl.show()

    # plot currents
    # pl.figure()
    # colors = c_map(np.linspace(0, 1, len(currents[0])))
    # for j, step_amp in enumerate(step_amps):
    #     for k, current in enumerate(currents[j]):
    #         pl.plot(t, -1*current, c=colors[k], label=channel_list[k])
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Current (mA/cm$^2$)')
    # pl.xlim(595, 645)
    # pl.tight_layout()
    pl.show()

    # # compare with data
    # for p_idx in [0, 1, 2, 3]:
    #     if p_idx == 0:
    #         v_data, t_data = get_v_and_t_from_heka(data_dir, 'hyperRampTester')
    #     else:
    #         v_data_tmp, t_data_tmp = get_v_and_t_from_heka(data_dir, 'hyperRampTester'+'('+str(p_idx)+')')
    #         v_data = np.concatenate((v_data, v_data_tmp), axis=0)
    #         t_data = np.concatenate((t_data, t_data_tmp), axis=0)
    # for p_idx in [0, 1, 2, 3]:
    #     if p_idx == 0:
    #         v_data_tmp, t_data_tmp = get_v_and_t_from_heka(data_dir, 'depoRampTester')
    #     else:
    #         v_data_tmp, t_data_tmp = get_v_and_t_from_heka(data_dir, 'depoRampTester'+'('+str(p_idx)+')')
    #     v_data = np.concatenate((v_data, v_data_tmp), axis=0)
    #     t_data = np.concatenate((t_data, t_data_tmp), axis=0)
    #
    # c_map = pl.cm.get_cmap('plasma')
    # colors = c_map(np.linspace(0, 1, len(step_amps)))
    # pl.figure(figsize=(8, 6))
    # for j, step_amp in enumerate(step_amps):
    #     pl.plot(t, v[j], c=colors[j], label='%.2f (nA)' % step_amp)
    # for k in range(len(v_data)):
    #     pl.plot(t_data[k], v_data[k]-8, 'k')
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Membrane potential (mV)')
    # pl.legend()
    # pl.xlim(595, 645)
    # pl.ylim(-95, -40)
    # pl.tight_layout()
    # pl.show()
    #pl.close()


def get_v_traces_t_traces_and_currents(cell, dt, ramp_amp, step_amps, tstop):
    currents = []
    v_traces = []
    t_traces = []
    for j, step_amp in enumerate(step_amps):
        i_inj = get_i_inj_hyper_depo_ramp(step_amp=step_amp, ramp_amp=ramp_amp, tstop=tstop, dt=dt)
        simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': -75, 'tstop': tstop,
                             'dt': dt, 'celsius': 35, 'onset': 200}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)
        v_traces.append(v)
        t_traces.append(t)
        currents_tmp, channel_list = simulate_currents(cell, simulation_params, plot=False)
        currents.append(currents_tmp)
    return t_traces, v_traces, currents


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'
    model_ids = range(1, 7)
    mechanism_dir = '../../../model/channels/vavoulis'

    save_dir = '../../../results/server_17_12_04/2017-12-16_10:04:51/'
    model_ids = [20]
    method = 'L-BFGS-B'

    load_mechanism_dir(mechanism_dir)
    for model_id in model_ids:
        # load model
        model_dir = os.path.join(save_dir, str(model_id), 'cell.json')
        model_dir = os.path.join(save_dir, str(model_id), method, 'cell.json')
        cell = Cell.from_modeldir(model_dir)

        # simulate
        simulate_hyper_depo_ramp(cell, save_dir=os.path.join(save_dir, str(model_id)))