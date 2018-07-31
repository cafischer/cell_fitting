from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell
import os
import json
from cell_fitting.optimization.evaluation.plot_sine_stimulus import simulate_sine_stimulus
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # apply stim
    amp1 = 0.4  # 0.5
    amp2 = 0.2  # 0.2
    freq1 = 0.2  # 0.5: 1000, 0.25: 2000, 0.1: 5000, 0.05: 10000
    sine1_dur = 1./freq1 * 1000 / 2
    freq2 = 5  # 5  # 20
    onset_dur = offset_dur = 500
    dt = 0.01
    sine_params = {'amp1': amp1, 'amp2': amp2, 'sine1_dur': sine1_dur, 'freq2': freq2, 'onset_dur': onset_dur,
                   'offset_dur': offset_dur, 'dt': dt}

    v, t, i_inj = simulate_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'sine_stimulus', 'traces',
                                str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'v.npy'), v)
    np.save(os.path.join(save_dir_img, 't.npy'), t)
    np.save(os.path.join(save_dir_img, 'i_inj.npy'), i_inj)
    with open(os.path.join(save_dir_img, 'sine_params.json'), 'w') as f:
        json.dump(sine_params, f)

    print save_dir_img
    pl.figure()
    #pl.title('amp1: ' + str(amp1) + ', amp2: ' + str(amp2) + ', sine1dur: ' + str(sine1_dur) + ', freq2: ' + str(freq2), fontsize=16)
    pl.plot(t, v, 'r', linewidth=1)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'v.png'))
    pl.show()

    # pl.figure()
    # pl.title('amp1: ' + str(amp1) + ', amp2: ' + str(amp2) + ', sine1dur: ' + str(sine1_dur) + ', freq2: ' + str(freq2), fontsize=16)
    # pl.plot(t, v, 'r', linewidth=1)
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Membrane Potential (mV)')
    # pl.xlim(4000, 6000)
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'v_zoom.png'))
    # pl.show()

    # # plot influence of input current
    # from cell_fitting.optimization.helpers import *
    # L = cell.soma.L  # um
    # diam = cell.soma.diam  # um
    # cm = cell.soma.cm  # uF/cm**2
    # dt = t[1] - t[0]  # ms
    # dvdt = np.concatenate((np.array([(v[1] - v[0]) / dt]), np.diff(v) / dt))  # V
    #
    # # convert units
    # cell_area = get_cellarea(convert_from_unit('u', L),
    #                          convert_from_unit('u', diam))  # m**2
    # Cm = convert_from_unit('c', cm) * cell_area  # F
    # i_inj_c = convert_from_unit('n', np.array(i_inj))  # A
    #
    # i_ion = -1 * (dvdt * Cm - i_inj_c)  # A
    #
    # simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': -75, 'tstop': sine1_dur+1000,
    #                      'dt': dt, 'celsius': 35, 'onset': 200}
    # from cell_fitting.optimization.simulate import simulate_currents
    # currents, channel_list = simulate_currents(cell, simulation_params)
    #
    # i_ion_from_currents = 10 * np.sum(currents) * cell_area
    #
    # pl.figure()
    # #pl.title('amp1: ' + str(amp1) + ', amp2: ' + str(amp2) + ', sine1dur: ' + str(sine1_dur) + ', freq2: ' + str(freq2))
    # pl.plot(t, dvdt * Cm, 'k', label='$c_m dV/dt$')
    # pl.plot(t, i_ion, 'r', label='$I_{ion}$')
    # #pl.plot(t, i_ion_from_currents, 'k', label='$I_{ion} from currents$')
    # pl.plot(t, i_inj_c, 'b', label='$I_{inj}$')
    # pl.hlines(0, 1850, 2050, colors='0.5', linestyles='-')
    # pl.ylim(-1.5*1e-9, 1.5*1e-9)
    # pl.xlim(1850, 2050)
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Current (A)')
    # pl.legend()
    # pl.tight_layout()
    # pl.show()
    #
    # # plot currents
    # pl.figure()
    # for c, current in enumerate(currents):
    #     pl.plot(t, -1*current, label=channel_list[c])
    # pl.hlines(0, 1850, 2050, colors='0.5', linestyles='-')
    # pl.xlim(1850, 2050)
    # pl.ylim(-0.15, 0.15)
    # pl.legend()
    # pl.tight_layout()
    # pl.show()
    #
    # channel_list = np.array(channel_list)
    # pl.figure()
    # pl.plot(t, -1 * currents[channel_list == 'nat'][0] + -1*currents[channel_list == 'kdr'][0]+-1 * currents[channel_list == 'pas'][0], label='nat + kdr + pas')
    # pl.plot(t, -1 * currents[channel_list == 'nap'][0] + -1*currents[channel_list == 'kdr'][0]+-1 * currents[channel_list == 'pas'][0], label='nap + kdr + pas')
    # pl.plot(t, -1 * currents[channel_list == 'pas'][0], label='pas')
    # pl.hlines(0, 1850, 2050, colors='0.5', linestyles='-')
    # pl.xlim(1850, 2050)
    # pl.ylim(-0.15, 0.15)
    # pl.legend()
    # pl.tight_layout()
    # pl.show()