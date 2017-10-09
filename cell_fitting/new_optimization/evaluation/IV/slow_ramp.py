from __future__ import division
import matplotlib.pyplot as pl
pl.style.use('paper')
import os
from nrn_wrapper import Cell
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.new_optimization.evaluation.IV import get_IV, get_slow_ramp, get_slow_ramp_reverse

__author__ = 'caro'


if __name__ == '__main__':
    # parameters
    #save_dir = '../../../results/server/2017-07-27_09:18:59/22/L-BFGS-B'
    #model_dir = os.path.join(save_dir, 'model', 'cell.json')
    save_dir = '../../../results/hand_tuning/test0'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # slow ramps
    step_st_ms = 100  # ms
    step_end_ms = 2900  # ms
    tstop = 3000  # ms
    dt = 0.001

    save_img = os.path.join(save_dir, 'img', 'IV', 'slow_ramp')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    step_amp_down = 0.0
    v_down, t_down, i_inj_down = get_IV(cell, step_amp_down, get_slow_ramp_reverse, step_st_ms, step_end_ms, tstop,
                                        v_init=-75, dt=dt)

    step_amp_up = 1.0
    v_up, t_up, i_inj_up = get_IV(cell, step_amp_up, get_slow_ramp, step_st_ms, step_end_ms, tstop,
                                        v_init=-75, dt=dt)

    onset_dur = offset_dur = 200
    amps_down = lambda x: (1 - step_amp_down) / (onset_dur - t_down[-1] - offset_dur) * x \
                          + (1-(1-step_amp_down)/(onset_dur-t_down[-1]-offset_dur)*onset_dur)
    heavy_amps_down = lambda x: amps_down(x) if onset_dur < x < t_down[-1] - offset_dur else 0
    amps_down_out = lambda x: "%.2f" % heavy_amps_down(x)

    amps_up = lambda x: (0 - step_amp_up) / (onset_dur - t_up[-1] - offset_dur) * x \
                          + (0-(0-step_amp_up)/(onset_dur-t_up[-1]-offset_dur)*onset_dur)
    heavy_amps_up = lambda x: amps_up(x) if onset_dur < x < t_up[-1] - offset_dur else 0
    amps_up_out = lambda x: "%.2f" % heavy_amps_down(x)


    # plot
    fig, ax = pl.subplots(2, 1, figsize=(10.667, 8))

    # v
    ax[0].plot(t_down, v_down, 'r')
    ax[1].plot(t_up, v_up, 'r')

    # labels
    ax[0].set_ylabel('Membrane \npotential (mV)')
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_xlim(t_down[-1], 0)
    ax02 = ax[0].twiny()
    ax02.set_xticks(t_down[int(round(100/dt))::int(round(560/dt))])
    ax02.set_xticklabels(["%.2f" % i for i in i_inj_down[int(round(100/dt))::int(round(560/dt))]])
    ax02.set_xlim(t_down[-1], 0)
    ax02.set_xlabel('Amplitude (nA)')

    ax[1].set_ylabel('Membrane \npotential (mV)')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_xlim(0, t_down[-1])
    ax[1].set_ylim(ax[0].get_ylim()[0],ax[0].get_ylim()[1])
    ax12 = ax[1].twiny()
    ax12.set_xticks(t_down[int(round(100/dt))::int(round(560/dt))])
    ax12.set_xticklabels(["%.2f" % i for i in i_inj_down[int(round(100/dt))::int(round(560/dt))][::-1]])
    ax12.set_xlim(0, t_down[-1])
    ax12.set_xlabel('Amplitude (nA)')

    # vertical lines
    last_spike_down = get_AP_onset_idxs(v_down, threshold=-30)[-1]
    t_last_spike_down = t_down[last_spike_down]
    onset_idxs_up = get_AP_onset_idxs(v_up, threshold=-30)
    first_spike_up = onset_idxs_up[0] if len(onset_idxs_up) > 0 else int(round(step_end_ms/dt))
    t_first_spike_up = t_up[first_spike_up]
    if i_inj_up[first_spike_up] > i_inj_down[last_spike_down]:
        ax[0].axvline(t_last_spike_down, ax[0].get_ylim()[0], ax[0].get_ylim()[1], linestyle='--', linewidth=1, alpha=0.5)
        ax[0].axvline(t_down[-1] - t_first_spike_up, ax[0].get_ylim()[0], ax[0].get_ylim()[1], linestyle='--', linewidth=1, alpha=0.5)
        ax[1].axvline(t_up[-1] - t_last_spike_down, ax[1].get_ylim()[0], ax[1].get_ylim()[1], linestyle='--', linewidth=1, alpha=0.5)
        ax[1].axvline(t_first_spike_up, ax[1].get_ylim()[0], ax[1].get_ylim()[1], linestyle='--', linewidth=1, alpha=0.5)

    pl.tight_layout()
    pl.savefig(os.path.join(save_img, 'slow_ramp.png'))
    pl.show()


    # plot stimulus
    fig, ax = pl.subplots(2, 1, figsize=(10.667, 8))

    # v
    ax[0].plot(t_down, i_inj_down, 'r')
    ax[1].plot(t_up, i_inj_up, 'r')

    # labels
    ax[0].set_ylabel('Current (nA)')
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_xlim(t_down[-1], 0)

    ax[1].set_ylabel('Current (nA)')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_xlim(0, t_down[-1])
    ax[1].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1])

    pl.tight_layout()
    pl.savefig(os.path.join(save_img, 'slow_ramp_i_inj.png'))
    pl.show()