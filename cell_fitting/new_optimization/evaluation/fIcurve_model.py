import matplotlib.pyplot as pl
import numpy as np
import matplotlib
import os
from cell_fitting.new_optimization.fitter import iclamp_handling_onset
from nrn_wrapper import Cell
from cell_characteristics.fIcurve import compute_fIcurve

__author__ = 'caro'


def get_step(start_idx, end_idx, len_idx, step_amp):
    i_exp = np.zeros(len_idx)
    i_exp[start_idx:end_idx] = step_amp
    return i_exp

def get_slow_ramp(start_idx, end_idx, len_idx, step_amp):
    i_exp = np.zeros(len_idx)
    i_exp[start_idx:end_idx] = np.linspace(0, step_amp, end_idx-start_idx)
    return i_exp

def get_slow_ramp_reverse(start_idx, end_idx, len_idx, step_amp):
    i_exp = np.zeros(len_idx)
    i_exp[start_idx:end_idx] = np.linspace(1, step_amp, end_idx - start_idx)
    return i_exp


def get_IV(cell, step_amp, step_fun, step_st_ms, step_end_ms, tstop, v_init=-75, dt = 0.001):

    t_exp = np.arange(0, tstop + dt, dt)
    i_exp = step_fun(int(round(step_st_ms/dt)), int(round(step_end_ms/dt)), int(round(tstop/dt)), step_amp)

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init, 'tstop': tstop,
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    return v, t, i_exp


if __name__ == '__main__':
    # parameters
    save_dir = '../../results/server/2017-07-18_11:14:25/17/L-BFGS-B'
    model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../results/hand_tuning/test0/'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'


    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # high resolution FI-curve
    # step_st_ms = 200  # ms
    # step_end_ms = 800  # ms
    # tstop = 1000  # ms
    # step_amps = np.arange(0.16, 0.3, 0.005)
    # vs = []
    # i_injs = []
    # for step_amp in step_amps:
    #     v, t, i_inj = get_IV(cell, step_amp, v_init=-75)
    #     vs.append(v)
    #     i_injs.append((i_inj))
    #
    # vs = np.vstack(vs)
    # i_injs = np.vstack(i_injs)
    # amps, firing_rates = compute_fIcurve(vs, i_injs, t)
    #
    # plot
    #save_img = os.path.join(save_dir, 'img', 'IV', 'high_res')
    # if not os.path.exists(save_img):
    #     os.makedirs(save_img)
    #
    # cmap = matplotlib.cm.get_cmap('Reds')
    # colors = [cmap(x) for x in np.linspace(0.2, 1.0, len(vs))]
    # pl.figure()
    # for i, v in enumerate(vs):
    #     pl.plot(t, v, color=colors[i], label='Model' if i == 0 else None)
    # pl.xlabel('Time $(ms)$', fontsize=16)
    # pl.ylabel('Membrane potential $(mV)$', fontsize=16)
    # pl.legend(loc='upper right', fontsize=16)
    # #pl.savefig(os.path.join(save_img, 'IV.svg'))
    # pl.show()
    #
    # pl.figure()
    # for i, v in enumerate(vs):
    #     pl.plot(t, v, color=colors[i], label='Model' if i == 0 else None)
    # pl.xlabel('Time $(ms)$', fontsize=16)
    # pl.ylabel('Membrane potential $(mV)$', fontsize=16)
    # pl.xlim(200, 800)
    # pl.ylim(-75, -55)
    # pl.legend(loc='upper right', fontsize=16)
    # pl.savefig(os.path.join(save_img, 'IV_zoom1.svg'))
    #
    # pl.figure()
    # for i, v in enumerate(vs):
    #     pl.plot(t, v, color=colors[i], label='Model' if i == 0 else None)
    # pl.xlabel('Time $(ms)$', fontsize=16)
    # pl.ylabel('Membrane potential $(mV)$', fontsize=16)
    # pl.xlim(200, 300)
    # pl.ylim(-75, -55)
    # pl.legend(loc='upper right', fontsize=16)
    # pl.savefig(os.path.join(save_img, 'IV_zoom2.svg'))
    #
    # pl.figure()
    # pl.plot(amps, firing_rates, '-ok')
    # pl.ylabel('Firing rate (APs/ms)')
    # pl.xlabel('Current (nA)')
    # pl.savefig(os.path.join(save_img, 'fI_curve.svg'))
    # pl.show()


    # slow ramps
    step_st_ms = 200  # ms
    step_end_ms = 3800  # ms
    tstop = 4000  # ms
    dt = 0.001

    save_img = os.path.join(save_dir, 'img', 'IV', 'slow_ramp')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    step_amp_down = 0.3
    v_down, t_down, i_inj_down = get_IV(cell, step_amp_down, get_slow_ramp_reverse, step_st_ms, step_end_ms, tstop,
                                        v_init=-75, dt=dt)

    step_amp_up = 0.7
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

    fig, ax = pl.subplots(2, 1, figsize=(16, 10))
    ax[0].plot(t_down, v_down, 'k')
    ax[1].plot(t_up, v_up, 'k')

    ax[0].set_ylabel('Membrane \npotential $(mV)$', fontsize=16)
    ax[0].set_xlabel('Time $(ms)$', fontsize=16)
    ax[0].set_xlim(0, t_down[-1])
    ax02 = ax[0].twiny()
    ax02.set_xticks(t_down[::int(round(100/dt))])
    ax02.set_xticklabels(["%.2f" % i for i in i_inj_down[::int(round(100/dt))]])
    ax02.set_xlabel('Amplitude $(nA)$', fontsize=16)

    ax[1].set_ylabel('Membrane \npotential $(mV)$', fontsize=16)
    ax[1].set_xlabel('Time $(ms)$', fontsize=16)
    ax[1].set_xlim(0, t_down[-1])
    ax12 = ax[1].twiny()
    ax12.set_xticks(t_up[::int(round(100/dt))])
    ax12.set_xticklabels(["%.2f" % i for i in i_inj_up[::int(round(100/dt))]])
    ax12.set_xlabel('Amplitude $(nA)$', fontsize=16)

    pl.tight_layout()
    #pl.savefig(os.path.join(save_img, 'slow_ramp.png'))
    pl.show()
