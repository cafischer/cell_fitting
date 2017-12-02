from __future__ import division
import numpy as np
from cell_characteristics import to_idx


def get_ramp3_times(delta_first=3, delta_ramp=2, n_times=10):
    return np.arange(delta_first, n_times * delta_ramp + delta_ramp, delta_ramp)


def get_ramp(start, peak, end, amp_before, ramp_amp, amp_after, dt):
    start_idx = to_idx(start, dt)
    peak_idx = to_idx(peak, dt)
    end_idx = to_idx(end, dt, 4)
    diff_idx = end_idx - start_idx + 1
    half_diff_up = peak_idx - start_idx + 1
    half_diff_down = end_idx - peak_idx + 1
    i_inj = np.zeros(diff_idx)
    i_inj[:half_diff_up] = np.linspace(amp_before, ramp_amp, half_diff_up)
    i_inj[half_diff_up:] = np.linspace(ramp_amp, amp_after, half_diff_down)[1:]
    return i_inj


def get_i_inj_rampIV(ramp_start, ramp_peak, ramp_end, amp_before, ramp_amp, amp_after, tstop, dt):
    i_inj = np.zeros(to_idx(tstop, dt)+1)
    ramp_start_idx = to_idx(ramp_start, dt)
    ramp_end_idx = to_idx(ramp_end, dt)
    i_inj[:ramp_start_idx] = amp_before
    i_inj[ramp_start_idx:ramp_end_idx+1] = get_ramp(ramp_start, ramp_peak, ramp_end,
                                                    amp_before, ramp_amp, amp_after, dt)
    i_inj[ramp_end_idx:] = amp_after
    return i_inj


def get_i_inj_step(start_step, end_step, step_amp, tstop, dt):
    i_inj = np.zeros(to_idx(tstop, dt)+1)
    i_inj[to_idx(start_step, dt):to_idx(end_step, dt)] = step_amp
    return i_inj


def get_i_inj_zap(amp=0.1, freq0=0, freq1=20, onset_dur=2000, offset_dur=2000, zap_dur=30000, tstop=34000, dt=0.01):
    t = np.arange(0, zap_dur+dt/2, dt)
    assert onset_dur + offset_dur + zap_dur == tstop
    zap = amp * np.sin(2 * np.pi * ((freq1 - freq0) / 1000 * t / (2 * t[-1]) + freq0/1000) * t)
    onset = np.zeros(to_idx(onset_dur, dt, 4))
    offset = np.zeros(to_idx(offset_dur, dt, 4))
    zap_stim = np.concatenate((onset, zap, offset))
    return zap_stim


def get_i_inj_double_ramp(ramp_amp, ramp3_amp, ramp3_time, step_amp, len_step=250, baseline_amp=-0.05, len_ramp=2,
                            start_ramp1=20, start_step=222, len_step2ramp=15, tstop=500, dt=0.01):

    end_ramp1 = start_ramp1 + len_ramp
    start_ramp2 = start_step + len_step + len_step2ramp
    end_ramp2 = start_ramp2 + len_ramp
    start_ramp3 = end_ramp2 + ramp3_time

    start_ramp1_idx = to_idx(start_ramp1, dt)
    end_ramp1_idx = to_idx(end_ramp1, dt)
    start_step_idx = to_idx(start_step, dt)
    end_step_idx = start_step_idx + to_idx(len_step, dt)
    start_ramp2_idx = to_idx(start_ramp2, dt)
    end_ramp2_idx = to_idx(end_ramp2, dt)
    start_ramp3_idx = to_idx(start_ramp3, dt)
    end_ramp3_idx = start_ramp3_idx + to_idx(len_ramp, dt)

    i_inj = np.ones(to_idx(tstop, dt)+1) * baseline_amp
    i_inj[start_ramp1_idx:end_ramp1_idx+1] = get_ramp(start_ramp1, start_ramp1 + 0.8, start_ramp1 + len_ramp,
                                                      baseline_amp, ramp_amp, baseline_amp, dt)
    i_inj[start_step_idx:end_step_idx] = step_amp
    i_inj[start_ramp2_idx:end_ramp2_idx+1] = get_ramp(start_ramp2, start_ramp2 + 0.8, start_ramp2 + len_ramp,
                                                      baseline_amp, ramp_amp, baseline_amp, dt)
    i_inj[start_ramp3_idx:end_ramp3_idx+1] = get_ramp(start_ramp3, start_ramp3 + 0.8, start_ramp3 + len_ramp,
                                                      baseline_amp, ramp3_amp, baseline_amp, dt)
    return i_inj


def get_i_inj_hyper_depo_ramp(step_start=200, step_end=600, ramp_len=2, step_amp=-0.2, ramp_amp=5, tstop=1000, dt=0.01):

    ramp_end = step_end + ramp_len
    step_st_idx = to_idx(step_start, dt)
    step_end_idx = to_idx(step_end, dt)
    ramp_end_idx = to_idx(ramp_end, dt)

    i_inj = np.zeros(to_idx(tstop, dt)+1)
    i_inj[step_st_idx:step_end_idx] = step_amp
    i_inj[step_end_idx:ramp_end_idx+1] = get_ramp(step_end, step_end + 0.8, ramp_end, step_amp, ramp_amp, 0, dt)
    return i_inj