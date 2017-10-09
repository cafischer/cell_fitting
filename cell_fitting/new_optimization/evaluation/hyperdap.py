import pylab as pl
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import simulate_currents, iclamp_handling_onset
pl.style.use('paper')

__author__ = 'caro'


def get_ramp(start_idx, end_idx, amp_before, ramp_amp, amp_after):
    diff_idx = end_idx - start_idx
    half_diff_up = int(round(diff_idx / 2)) - 1
    half_diff_down = int(round(diff_idx / 2)) + 1  # peak is one earlier
    if diff_idx % 2 != 0:
        half_diff_down += 1
    i_exp = np.zeros(diff_idx)
    i_exp[:half_diff_up] = np.linspace(amp_before, ramp_amp, half_diff_up)
    i_exp[half_diff_up:] = np.linspace(ramp_amp, amp_after, half_diff_down+1)[1:]
    return i_exp


def hyperpolarize_ramp(cell):
    """
    params:
    hyperamps = np.arange(-0.25, 0.26, 0.05)  # nA
    ramp_amp = 8  # nA
    dt = 0.01
    hyp_st_ms = 200  # ms
    hyp_end_ms = 600  # ms
    ramp_end_ms = 602  # ms
    tstop = 1000  # ms
    """

    #hyperamps = np.arange(-0.25, 0.26, 0.05)  # nA  #TODO
    hyperamps = np.arange(-0.2, 0.21, 0.05)
    ramp_amp = 8  # nA
    dt = 0.01
    hyp_st_ms = 200  # ms
    hyp_end_ms = 600  # ms
    ramp_end_ms = 602  # ms
    tstop = 1000  # ms

    hyp_st = int(round(hyp_st_ms / dt))
    hyp_end = int(round(hyp_end_ms / dt))
    ramp_end = int(round(ramp_end_ms / dt)) + 1

    t_exp = np.arange(0, tstop + dt, dt)

    v = np.zeros([len(hyperamps), len(t_exp)])
    currents = []
    for j, hyper_amp in enumerate(hyperamps):
        i_exp = np.zeros(len(t_exp))
        i_exp[hyp_st:hyp_end] = hyper_amp
        i_exp[hyp_end:ramp_end] = get_ramp(hyp_end, ramp_end, hyper_amp, ramp_amp, 0)

        # get simulation parameters
        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -59, 'tstop': t_exp[-1],
                             'dt': dt, 'celsius': 35, 'onset': 200}

        # record v
        v[j], t, _ = iclamp_handling_onset(cell, **simulation_params)

        currents_tmp, channel_list = simulate_currents(cell, simulation_params, plot=False)
        currents.append(currents_tmp)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'hyperdap')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    c_map = pl.cm.get_cmap('plasma')
    colors = c_map(np.linspace(0, 1, len(hyperamps)))

    pl.figure(figsize=(8, 6))
    for j, hyper_amp in enumerate(hyperamps):
        pl.plot(t, v[j], c=colors[j], label=str(np.round(hyper_amp, 2)) + ' nA')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend(loc='upper right')
    pl.tight_layout()
    pl.xlim(0, t[-1])
    pl.savefig(os.path.join(save_dir_img, 'hyperDAP.png'))
    pl.show()

    pl.figure(figsize=(8, 6))
    for j, hyper_amp in enumerate(hyperamps):
        pl.plot(t, v[j], c=colors[j], label=str(np.round(hyper_amp, 2)) + ' nA')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.xlim(595, 645)
    pl.ylim(-95, -40)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'hyperDAP_zoom.png'))
    pl.show()

    # # plot currents
    pl.figure()
    colors = c_map(np.linspace(0, 1, len(currents[0])))
    for j, hyper_amp in enumerate(hyperamps):
        for k, current in enumerate(currents[j]):
            pl.plot(t, -1*current, c=colors[k], label=channel_list[k])
    pl.xlabel('Time (ms)')
    pl.ylabel('Current (mA/cm$^2$)')
    pl.xlim(595, 645)
    pl.tight_layout()
    pl.show()



if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    # save_dir = '../../results/server/2017-07-18_11:14:25/17/L-BFGS-B/'
    # model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../results/hand_tuning/cell_2017-07-24_13:59:54_21_0/'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    print cell.soma(.5).nat.h_vs
    cell.soma(.5).nat.h_vs = -15.14


    # start hyperdap
    hyperpolarize_ramp(cell)