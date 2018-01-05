import os
import numpy as np
import pylab as pl
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict, get_characteristic_name_dict, \
    get_characteristic_unit_dict
from cell_fitting.optimization.evaluation.plot_hyper_depo import simulate_hyper_depo, \
    get_spike_characteristics_and_vstep, plot_hyper_depo
pl.style.use('paper')

__author__ = 'caro'


def apply_hyper_depo_ramp(cell, save_dir):
    step_amps = np.array([-0.25, -0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.25])
    ramp_amp = 5.0  # nA
    step_start = 200
    ramp_start = 600
    dt = 0.01
    tstop = 1000  # ms
    spike_characteristic_params = get_spike_characteristics_dict()
    return_characteristics = ['DAP_amp', 'DAP_deflection', 'DAP_width', 'fAHP_amp', 'DAP_time']

    t_traces, v_traces, currents = simulate_hyper_depo(cell, dt, ramp_amp, step_amps, tstop)

    spike_characteristics_mat, v_step = get_spike_characteristics_and_vstep(v_traces, t_traces,
                                                                            spike_characteristic_params,
                                                                            return_characteristics, ramp_start,
                                                                            step_start)

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

    plot_hyper_depo(step_amps, t_traces, v_traces, save_dir_img)

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
        pl.ylabel('$RMSE_{%s}$ (%s)' % (get_characteristic_name_dict()[return_characteristics[i]].replace(' ', '\ '),
                                               get_characteristic_unit_dict()[return_characteristics[i]]))
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



if __name__ == '__main__':
    # parameters
    #save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'
    #model_ids = range(1, 7)
    mechanism_dir = '../../../model/channels/vavoulis'

    save_dir = '../../../results/server_17_12_04/2017-12-26_08:17:19'
    model_ids = [473]
    method = 'L-BFGS-B'

    load_mechanism_dir(mechanism_dir)
    for model_id in model_ids:
        model_dir = os.path.join(save_dir, str(model_id), 'cell.json')
        model_dir = os.path.join(save_dir, str(model_id), method, 'cell.json')
        cell = Cell.from_modeldir(model_dir)

        apply_hyper_depo_ramp(cell, save_dir=os.path.join(save_dir, str(model_id)))