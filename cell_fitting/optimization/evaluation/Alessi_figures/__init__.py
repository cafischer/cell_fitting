from __future__ import division

import matplotlib.pyplot as pl
import numpy as np
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs

from cell_fitting.optimization.evaluation.plot_IV import get_step, get_IV
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.util import init_nan

pl.style.use('paper')


def find_hold_amps(cell, hold_potentials, test_hold_amps, tstop, dt, plot=False):
    hold_amp_last = hold_potentials[0]
    hold_amps = init_nan(len(hold_potentials))
    for i, hold_potential in enumerate(hold_potentials):
        for hold_amp in test_hold_amps[test_hold_amps > hold_amp_last]:
            v, t, i_inj = get_IV(cell, hold_amp, get_step, 0, tstop, tstop, v_init=hold_potential, dt=dt)
            if np.round(np.mean(v), 1) >= np.round(hold_potential, 1):
                if np.round(np.mean(v), 1) > np.round(hold_potential, 1):
                    print np.mean(v)  # TODO
                if np.round(np.mean(v), 1) == np.round(hold_potential, 1):
                    hold_amps[i] = hold_amp
                    hold_amp_last = hold_amp
                    if plot:
                        pl.figure()
                        pl.plot(t, v)
                        pl.show()
                    break
        print hold_amp
    return hold_amps


def find_AP_current(cell, other_i_inj, test_step_amps, step_st_ms, step_end_ms, AP_threshold, v_init, tstop, dt,
                    onset=200, celsius=35, step_to_AP_dur=5, plot=False):
    step_amp_spike = None
    for step_amp in test_step_amps:
        i_step = get_step(to_idx(step_st_ms, dt), to_idx(step_end_ms, dt), to_idx(tstop, dt) + 1, step_amp)
        i_exp = other_i_inj + i_step

        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init,
                             'tstop': tstop, 'dt': dt, 'celsius': celsius, 'onset': onset}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)

        onsets = get_AP_onset_idxs(v, AP_threshold)
        onsets_after_step = onsets[np.logical_and(to_idx(step_st_ms, dt) < onsets,
                                     onsets < to_idx(step_st_ms + step_to_AP_dur, dt))]

        if len(onsets_after_step) > 0:
            step_amp_spike = step_amp
            print 'Step current amplitude: %.2f' % step_amp_spike
            break

        if plot:
            pl.figure()
            pl.plot(t, v)
            pl.show()

    return step_amp_spike


def simulate_with_step_and_holding_current(cell, hold_potentials, hold_amps, step_amp_spike, step_st_ms, step_end_ms,
                                           tstop, dt, onset=200, celsius=35):
    v_mat = []
    for i, hold_amp in enumerate(hold_amps):
        i_step = get_step(to_idx(step_st_ms, dt), to_idx(step_end_ms, dt), to_idx(tstop, dt) + 1, step_amp_spike)
        i_hold = get_step(0, to_idx(tstop, dt) + 1, to_idx(tstop, dt) + 1, hold_amp)
        i_exp = i_hold + i_step

        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': hold_potentials[i],
                             'tstop': tstop, 'dt': dt, 'celsius': celsius, 'onset': onset}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)
        v_mat.append(v)
    return v_mat, t