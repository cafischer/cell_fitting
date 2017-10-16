from __future__ import division
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.new_optimization.evaluation.Alessi_figures import find_hold_amps
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.new_optimization.evaluation.IV import get_step
from cell_fitting.new_optimization.evaluation.increase_rampIV import get_ramp
from cell_fitting.util import init_nan
import matplotlib.pyplot as pl
pl.style.use('paper')

__author__ = 'caro'


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    AP_amp_tol = 1
    AP_width_tol = 0.1
    AP_time_tol = 0.1
    percent_block = 0.9
    use_same_hold_amps = False
    use_given_hold_amps_block = True
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
    test_hold_amps = np.arange(-1, 10, 0.001)
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
    elif use_given_hold_amps_block:
        hold_amps_block = np.load(os.path.join(save_dir_img, 'hold_amps_block.npy'))
    else:
        hold_amps_block = find_hold_amps(cell, hold_potentials, test_hold_amps, tstop, dt, plot=False)
        print 'Holding amplitudes: ' + str(hold_amps_block)

    # reconstruct AP
    v_mat_block = []
    ramp_amps = init_nan(len(hold_potentials))
    ramp_durs = init_nan(len(hold_potentials))
    ramp_shifts = init_nan(len(hold_potentials))
    for i, (hold_amp, hold_potential) in enumerate(zip(hold_amps_block, hold_potentials)):
        if np.isnan(hold_amp):
            print 'Hold amp is nan!'
            v_mat_block.append(init_nan(len(t)))
            break

        AP_amp, AP_width, AP_time = get_spike_characteristics(v_mat[i], t, ['AP_amp', 'AP_width', 'AP_time'],
                                                              hold_potential, AP_interval=4, check=False)
        AP_amp_block = AP_amp - 2 * AP_amp_tol
        AP_width_block = AP_width - 2 * AP_width_tol
        AP_time_block = AP_time - 2 * AP_time_tol

        step_dur = step_end_ms - step_st_ms
        ramp_amp = 0
        ramp_dur = 0.1
        ramp_shift = 0
        counter = 0
        count_max = 1000
        AP_width_None = True
        AP_time_None = True
        while np.abs(AP_amp - AP_amp_block) > AP_amp_tol or np.abs(AP_width - AP_width_block) > AP_width_tol\
                or np.abs(AP_time - AP_time_block) > AP_time_tol:
            if counter > count_max:
                break
            counter += 1
            if np.abs(AP_amp - AP_amp_block) > AP_amp_tol:
                if np.abs(AP_amp - AP_amp_block) > 10:
                    if AP_amp_block < AP_amp:
                        ramp_amp += 1.0
                    else:
                        ramp_amp -= 1.0
                elif np.abs(AP_amp - AP_amp_block) > 2:
                    if AP_amp_block < AP_amp:
                        ramp_amp += 0.1
                    else:
                        ramp_amp -= 0.1
                else:
                    if AP_amp_block < AP_amp:
                        ramp_amp += 0.01
                    else:
                        ramp_amp -= 0.01
            if not AP_width_None:
                if np.abs(AP_width - AP_width_block) > AP_width_tol:
                    if np.abs(AP_width - AP_width_block) > 2:
                        if AP_width_block < AP_width:
                            ramp_dur += 0.2
                        else:
                            ramp_dur -= 0.2
                    else:
                        if AP_width_block < AP_width:
                            ramp_dur += 0.01
                        else:
                            ramp_dur -= 0.01
            if ramp_dur <= 0:
                ramp_dur = 0.1
            if not AP_time_None:
                if np.abs(AP_time - AP_time_block) > AP_time_tol:
                    if np.abs(AP_time - AP_time_block) > 1:
                        if AP_time_block < AP_time:
                            ramp_shift += 0.5
                        else:
                            ramp_shift -= 0.5
                    else:
                        if AP_time_block < AP_time:
                            ramp_shift += 0.01
                        else:
                            ramp_shift -= 0.01

            i_hold = get_step(0, to_idx(tstop, dt) + 1, to_idx(tstop, dt) + 1, hold_amp)
            i_step = get_step(to_idx(step_st_ms, dt), to_idx(step_st_ms + step_dur, dt, 6), to_idx(tstop, dt) + 1,
                              step_amp_spike)
            i_ramp = np.zeros(to_idx(tstop, dt) + 1)
            ramp_middle = AP_time + ramp_shift
            i_ramp[to_idx(ramp_middle-ramp_dur/2, dt, 6):to_idx(ramp_middle+ramp_dur/2, dt, 6)] = get_ramp(
                                                                to_idx(ramp_middle-ramp_dur/2, dt, 6),
                                                                to_idx(ramp_middle, dt, 6),
                                                                to_idx(ramp_middle+ramp_dur/2, dt, 6), 0, ramp_amp, 0)
            i_exp = i_hold + i_step + i_ramp
            simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': hold_potential,
                                 'tstop': tstop, 'dt': dt, 'celsius': celsius, 'onset': onset}
            v, t, _ = iclamp_handling_onset(cell, **simulation_params)
            AP_amp_block, AP_width_block, AP_time_block = get_spike_characteristics(v, t,
                                                                                    ['AP_amp', 'AP_width', 'AP_time'],
                                                                                    hold_potential, AP_interval=4,
                                                                                    check=False)

            print 'ramp amp: ' + str(ramp_amp)
            print 'ramp_dur: '+str(ramp_dur)
            print 'ramp_shift: ' + str(ramp_shift)
            print 'AP amp (nb/b): ' + str(AP_amp) + ' / ' + str(AP_amp_block)
            print 'AP width (nb/b): ' + str(AP_width) + ' / ' + str(AP_width_block)
            print 'AP time (nb/b): ' + str(AP_time) + ' / ' + str(AP_time_block)

            if AP_amp_block is None:
                AP_amp_block = AP_amp - AP_amp_tol - 10
            if AP_width_block is None:
                AP_width_block = AP_width - 2 * AP_width_tol
                AP_width_None = True
            else:
                AP_width_None = False
            if AP_time_block is None:
                AP_time_block = AP_time - 2 * AP_time_tol
                AP_time_None = True
            else:
                AP_time_None = False
            # pl.figure()
            # pl.plot(t, v_mat[i], 'k', label='without block')
            # pl.plot(t, v, 'r', label='with block')
            # pl.show()
        if counter <= count_max:
            v_mat_block.append(v)
            ramp_amps[i] = ramp_amp
            ramp_durs[i] = ramp_dur
            ramp_shifts[i] = ramp_shift
        else:
            print 'Could not reproduce AP!'
            v_mat_block.append(init_nan(len(t)))

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
    np.save(os.path.join(save_dir_img, 'ramp_amps.npy'), ramp_amps)
    np.save(os.path.join(save_dir_img, 'ramp_durs.npy'), ramp_durs)
    np.save(os.path.join(save_dir_img, 'ramp_shifts.npy'), ramp_shifts)

    for i, hold_potential in enumerate(hold_potentials):
        pl.figure()
        pl.plot(t, v_mat[i], 'r', label='without block')
        pl.plot(t, v_mat_block[i], 'orange', label='with ' + str(percent_block*100) + '% block of Na$^+$')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane potential (mV)')
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'v_'+str(hold_potential)+'.png'))
        pl.show()