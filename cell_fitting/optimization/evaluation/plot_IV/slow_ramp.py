from __future__ import division
import matplotlib.pyplot as pl
import os
from nrn_wrapper import Cell
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.optimization.evaluation.plot_IV import get_IV, get_slow_ramp, get_slow_ramp_reverse
pl.style.use('paper_subplots')
__author__ = 'caro'


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_img = '/home/cf/Dropbox/thesis/figures_discussion'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # slow ramps
    step_st_ms = 100  # ms
    step_end_ms = 2900  # ms
    tstop = 3000  # ms
    dt = 0.01

    save_img = os.path.join(save_dir, 'img', 'IV', 'slow_ramp')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    step_amp_down = 0.0
    v_down, t_down, i_inj_down = get_IV(cell, step_amp_down, get_slow_ramp_reverse, step_st_ms, step_end_ms, tstop, dt,
                                        {'v_init': -75})

    step_amp_up = 1.0
    v_up, t_up, i_inj_up = get_IV(cell, step_amp_up, get_slow_ramp, step_st_ms, step_end_ms, tstop, dt, {'v_init': -75})

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
    fig, ax = pl.subplots(4, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [5, 1, 5, 1]})

    # v
    ax[0].plot(t_up/1000., v_up, 'k')
    ax[2].plot(t_down/1000., v_down, 'k')

    # labels
    ax[0].set_ylabel('Mem. pot. (mV)')
    #ax[0].set_xlabel('Time (s)')
    ax[0].set_xlim(0, t_up[-1]/1000.)
    ax[0].set_xticks([])
    ax[0].set_ylim(-80, 60)
    # ax12 = ax[0].twiny()
    # ax12.set_xticks(t_down[int(round(100/dt))::int(round(560/dt))])
    # ax12.set_xticklabels(["%.2f" % i for i in i_inj_down[int(round(100/dt))::int(round(560/dt))][::-1]])
    # ax12.set_xlim(0, t_down[-1])
    # ax12.set_xlabel('Current (nA)')
    # ax12.spines['top'].set_visible(True)

    ax[2].set_ylabel('Mem. pot. (mV)')
    #ax[2].set_xlabel('Time (s)')
    ax[2].set_xlim(t_down[-1]/1000., 0)
    ax[2].set_xticks([])
    ax[2].set_ylim(-80, 60)
    # ax02 = ax[1].twiny()
    # ax02.set_xticks(t_down[int(round(100/dt))::int(round(560/dt))])
    # ax02.set_xticklabels(["%.2f" % i for i in i_inj_down[int(round(100/dt))::int(round(560/dt))]])
    # ax02.set_xlim(t_down[-1], 0)
    # ax02.set_xlabel('Current (nA)')
    # ax02.spines['top'].set_visible(True)

    ax[1].plot(t_up/1000., i_inj_up, 'k', clip_on=False)
    ax[1].set_ylabel('Current (nA)')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_xlim(0, t_up[-1]/1000.)
    ax[1].set_ylim(0, 1)

    ax[3].plot(t_down/1000., i_inj_down, 'k', clip_on=False)
    ax[3].set_ylabel('Current (nA)')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_xlim(t_down[-1]/1000., 0)
    ax[3].set_ylim(0, 1)

    ax[0].get_yaxis().set_label_coords(-0.04, 0.5)
    ax[1].get_yaxis().set_label_coords(-0.04, 0.5)
    ax[2].get_yaxis().set_label_coords(-0.04, 0.5)
    ax[3].get_yaxis().set_label_coords(-0.04, 0.5)

    # vertical lines
    last_spike_down = get_AP_onset_idxs(v_down, threshold=-30)[-1]
    t_last_spike_down = t_down[last_spike_down]
    onset_idxs_up = get_AP_onset_idxs(v_up, threshold=-30)
    first_spike_up = onset_idxs_up[0] if len(onset_idxs_up) > 0 else int(round(step_end_ms/dt))
    t_first_spike_up = t_up[first_spike_up]
    if i_inj_up[first_spike_up] > i_inj_down[last_spike_down]:
        ax[0].axvline((t_up[-1] - t_last_spike_down)/1000., 0, 1, linestyle='--', color='0.5')
        ax[0].axvline(t_first_spike_up/1000., 0, 1, linestyle='--', color='0.5')
        ax[2].axvline(t_last_spike_down/1000., 0, 1, linestyle='--', color='0.5')
        ax[2].axvline((t_down[-1] - t_first_spike_up)/1000., 0, 1, linestyle='--', color='0.5')

    ax[0].text(-0.08, 0.95, 'A', transform=ax[0].transAxes, size=18, weight='bold')
    ax[2].text(-0.08, 0.95, 'B', transform=ax[2].transAxes, size=18, weight='bold')
    pl.tight_layout()
    pl.subplots_adjust(hspace=0.33, left=0.08, bottom=0.07)
    pl.savefig(os.path.join(save_img, 'slow_ramp.png'))
    pl.savefig(os.path.join(save_dir_img, 'slow_ramp.png'))
    pl.show()