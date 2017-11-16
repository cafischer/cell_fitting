from __future__ import division
import numpy as np
from cell_characteristics import to_idx


def get_i_inj_rampIV(ramp_start, ramp_peak, ramp_end, amp_before, ramp_amp, amp_after, tstop, dt):
    i_inj = np.zeros(to_idx(tstop, dt)+1)
    ramp_start_idx = to_idx(ramp_start, dt)
    ramp_end_idx = to_idx(ramp_end, dt)
    i_inj[:ramp_start_idx] = amp_before
    i_inj[ramp_start_idx:ramp_end_idx+1] = get_i_inj_ramp(ramp_start, ramp_peak, ramp_end,
                                                amp_before, ramp_amp, amp_after, dt)
    i_inj[ramp_end_idx:] = amp_after
    return i_inj


def get_i_inj_ramp(start, peak, end, amp_before, ramp_amp, amp_after, dt):
    start_idx = to_idx(start, dt)
    peak_idx = to_idx(peak, dt)
    end_idx = to_idx(end, dt)
    diff_idx = end_idx - start_idx + 1
    half_diff_up = peak_idx - start_idx + 1
    half_diff_down = end_idx - peak_idx + 1
    i_inj = np.zeros(diff_idx)
    i_inj[:half_diff_up] = np.linspace(amp_before, ramp_amp, half_diff_up)
    i_inj[half_diff_up:] = np.linspace(ramp_amp, amp_after, half_diff_down)[1:]
    return i_inj


def get_i_inj_step(start_step, end_step, step_amp, tstop, dt):
    i_inj = np.zeros(to_idx(tstop, dt)+1)
    i_inj[to_idx(start_step, dt):to_idx(end_step, dt)] = step_amp  # TODO check
    return i_inj


def get_zap(amp=0.1, freq0=0, freq1=20, onset_dur=2000, offset_dur=2000, zap_dur=30000, tstop=34000, dt=0.01):
    t = np.arange(0, zap_dur+dt/2, dt)
    assert onset_dur + offset_dur + zap_dur == tstop
    zap = amp * np.sin(2 * np.pi * ((freq1 - freq0) / 1000 * t / (2 * t[-1]) + freq0/1000) * t)
    onset = np.zeros(to_idx(onset_dur, dt, 4))
    offset = np.zeros(to_idx(offset_dur, dt, 4))
    zap_stim = np.concatenate((onset, zap, offset))
    return zap_stim