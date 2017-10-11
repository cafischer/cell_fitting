from __future__ import division
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.new_optimization.evaluation.Alessi_figures import find_hold_amps, \
    simulate_with_step_and_holding_current
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.new_optimization.evaluation.IV import get_step
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
pl.style.use('paper')

__author__ = 'caro'


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    percent_block = 0.2
    use_same_hold_amps = False
    if use_same_hold_amps:
        save_dir_img = os.path.join(save_dir, 'img', 'TTX', 'reconstructed_AP', 'same_amps',
                                    'percent_block' + str(percent_block))
    else:
        save_dir_img = os.path.join(save_dir, 'img', 'TTX', 'reconstructed_AP', 'new_amps',
                                    'percent_block'+str(percent_block))
    save_dir_hold = os.path.join(save_dir, 'img', 'DAP_at_different_holding_potentials')

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # load holding potentials and amplitudes
    celsius = 35
    onset = 200
    dt = 0.001
    step_st_ms = 50  # ms
    step_end_ms = step_st_ms + 1  # ms
    tstop = 240  # ms
    test_hold_amps = np.arange(-2, 2, 0.001)
    step_amp_spike = np.load(os.path.join(save_dir_hold, 'step_amp_spike.npy'))
    hold_potentials = np.load(os.path.join(save_dir_hold, 'hold_potentials.npy'))
    hold_amps = np.load(os.path.join(save_dir_hold, 'hold_amps.npy'))
    v_mat = np.load(os.path.join(save_dir_hold, 'v_mat.npy'))
    t = np.load(os.path.join(save_dir_hold, 't.npy'))

    # TTX application
    cell.soma(.5).nat.gbar = cell.soma(.5).nat.gbar * (1 - percent_block)
    cell.soma(.5).nap.gbar = cell.soma(.5).nap.gbar * (1 - percent_block)

    # find hold_amps
    if use_same_hold_amps:
        hold_amps_block = hold_amps
    else:
        hold_amps_block = find_hold_amps(cell, hold_potentials, test_hold_amps, tstop, dt, plot=False)
        print 'Holding amplitudes: ' + str(hold_amps_block)

    # reconstruct AP
    v_mat_block = []
    for i, (hold_amp, hold_potential) in enumerate(zip(hold_amps_block, hold_potentials)):

        AP_amp, AP_width = get_spike_characteristics(v_mat[i], t, ['AP_amp', 'AP_width'], hold_potential, AP_interval=4,
                                                     check=False)
        AP_amp_block = AP_amp - 2
        AP_width_block = AP_width - 0.2

        step_dur = step_end_ms - step_st_ms
        while np.abs(AP_amp - AP_amp_block) > 1 or np.abs(AP_width - AP_width_block) > 0.1:
            if AP_amp_block < AP_amp:
                step_amp_spike += 0.1
            else:
                step_amp_spike -= 0.1

            if AP_width_block < AP_width:
                step_dur += 0.1
            else:
                step_dur -= 0.1

            # ramp_amp = 0
            # ramp_st_ms = step_end_ms
            # ramp_dur = 0.1

            i_hold = get_step(0, to_idx(tstop, dt) + 1, to_idx(tstop, dt) + 1, hold_amp)
            i_step = get_step(to_idx(step_st_ms, dt), to_idx(step_st_ms + step_dur, dt), to_idx(tstop, dt) + 1, step_amp_spike)
            #i_ramp = get_step(to_idx(ramp_st_ms, dt), to_idx(ramp_st_ms+ramp_dur, dt), to_idx(tstop, dt) + 1, ramp_amp)
            i_exp = i_hold + i_step # + i_ramp
            simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': hold_potential,
                                 'tstop': tstop, 'dt': dt, 'celsius': celsius, 'onset': onset}
            v, t, _ = iclamp_handling_onset(cell, **simulation_params)
            AP_amp_block, AP_width_block = get_spike_characteristics(v, t, ['AP_amp', 'AP_width'], hold_potential,
                                                                     AP_interval=4, check=False)
            print 'AP amp (nb/b): ' + str(AP_amp) + ' / ' + str(AP_amp_block)
            print 'AP width (nb/b): ' + str(AP_width) + ' / ' + str(AP_width_block)
            if np.isnan(AP_amp_block):
                AP_amp_block = AP_amp - 2
            if np.isnan(AP_width_block):
                AP_width_block = AP_width - 0.2
            pl.figure()
            pl.plot(t, v_mat[i], 'k', label='without block')
            pl.plot(t, v, 'r', label='with block')
            pl.show()
        v_mat_block.append(v)

    # plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'v_mat.npy'), v_mat)
    np.save(os.path.join(save_dir_img, 'v_mat_block.npy'), v_mat_block)
    np.save(os.path.join(save_dir_img, 't.npy'), t)
    np.save(os.path.join(save_dir_img, 'hold_potentials.npy'), hold_potentials)
    np.save(os.path.join(save_dir_img, 'hold_amps.npy'), hold_amps)
    np.save(os.path.join(save_dir_img, 'hold_amps_block.npy'), hold_amps_block)
    np.save(os.path.join(save_dir_img, 'step_amp_spike.npy'), step_amp_spike)

    for i, hold_potential in enumerate(hold_potentials):
        pl.figure()
        pl.plot(t, v_mat[i], 'r', label='without block')
        pl.plot(t, v_mat_block[i], 'o', label='with ' + str(percent_block*100) + '% block of Na$^+$')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane potential (mV)')
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'v_'+str(hold_potential)+'.png'))
        pl.show()

    # TODO: Alessi could affect DAP with their ramp, why do they need it