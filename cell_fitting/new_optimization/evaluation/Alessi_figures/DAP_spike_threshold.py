from __future__ import division
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.new_optimization.evaluation.IV import get_step, get_IV
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_characteristics import to_idx
import matplotlib.pyplot as pl
pl.style.use('paper')


__author__ = 'caro'


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/1'
    model_dir = os.path.join(save_dir, 'cell.json')
    #save_dir = '../../../results/server/2017-07-27_09:18:59/22/L-BFGS-B
    #model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../../results/hand_tuning/test0'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    save_dir_hold = os.path.join(save_dir, 'img', 'DAP_at_different_holding_potentials')

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # load holding potentials and amps
    hold_potentials = np.load(os.path.join(save_dir_hold, 'hold_potentials.npy'))
    hold_amps = np.load(os.path.join(save_dir_hold, 'hold_amps.npy'))
    hold_potential = np.nan  # hold_potentials[-1]  #
    hold_amp = 0  # hold_amps[-1]  #
    v_init = hold_potential if not np.isnan(hold_potential) else -75

    # different holding potentials
    dt = 0.001
    first_step_st_ms = 50  # ms
    first_step_end_ms = first_step_st_ms + 1  # ms
    second_step_st_ms = first_step_st_ms + 15  # ms
    second_step_end_ms = second_step_st_ms + 1  # ms
    tstop = 240  # ms
    step_amps = np.arange(0, 3.0, 0.1)
    AP_threshold = -20

    # find right AP current 1st spike
    first_spike_amp = np.nan
    for step_amp in step_amps:
        i_step = get_step(int(round(first_step_st_ms / dt)), int(round(first_step_end_ms / dt)), int(round(tstop / dt)) + 1, step_amp)
        i_hold = get_step(0, int(round(tstop/dt)) + 1, int(round(tstop/dt))+1, hold_amp)
        i_exp = i_hold + i_step

        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init,
                             'tstop': tstop, 'dt': dt, 'celsius': 35, 'onset': 200}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)

        # pl.figure()
        # pl.plot(t, v)
        # pl.show()
        onsets = get_AP_onset_idxs(v, AP_threshold)
        onsets_after_stim = onsets[np.logical_and(to_idx(first_step_st_ms, dt) < onsets,
                                     onsets < to_idx(first_step_st_ms + 10, dt))]
        if len(onsets_after_stim) > 0:
            first_spike_amp = step_amp
            onset_first_spike = onsets_after_stim[0]
            print 'First step amplitude: %.2f' % first_spike_amp
            # pl.figure()
            # pl.plot(t, v)
            # pl.show()
            break

    # find right AP current 2nd spike
    second_spike_amp = np.nan
    for step_amp in step_amps:
        i_step = get_step(to_idx(first_step_st_ms, dt), to_idx(first_step_end_ms, dt), to_idx(tstop, dt) + 1, first_spike_amp)
        i_hold = get_step(0, to_idx(tstop, dt) + 1, to_idx(tstop, dt) + 1, hold_amp)
        i_step2 = get_step(to_idx(second_step_st_ms, dt), to_idx(second_step_end_ms, dt), to_idx(tstop, dt) + 1, step_amp)
        i_exp = i_hold + i_step + i_step2

        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init,
                             'tstop': tstop, 'dt': dt, 'celsius': 35, 'onset': 200}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)

        # pl.figure()
        # pl.plot(t, v)
        # pl.show()
        onsets = get_AP_onset_idxs(v, AP_threshold)
        if len(onsets[np.logical_and(onset_first_spike < onsets,
                                     onsets < onset_first_spike + to_idx(20, dt))]) > 0:
            second_spike_amp = step_amp
            print 'Second step amplitude: %.2f' % second_spike_amp
            # pl.figure()
            # pl.plot(t, v)
            # pl.show()
            break


    # plot
    save_img = os.path.join(save_dir, 'img', 'DAP_spike_threshold', 'hold_potential'+str(hold_potential))
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    np.save(os.path.join(save_img, 'first_spike_amp.npy'), first_spike_amp)
    np.save(os.path.join(save_img, 'second_spike_amp.npy'), second_spike_amp)

    pl.figure()
    pl.plot(t, v, color='r', label='Model')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_img, 'v.png'))
    #pl.show()

    pl.figure()
    pl.bar(0, first_spike_amp, color='r')
    pl.bar(1, second_spike_amp, color='r')
    pl.xticks([0, 1], ['1st Pulse', '2nd Pulse'])
    pl.xlabel('Time (ms)')
    pl.ylabel('Current (nA)')
    pl.ylim(step_amps[0], step_amps[-1])
    pl.tight_layout()
    pl.savefig(os.path.join(save_img, 'amps.png'))
    pl.show()