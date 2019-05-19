import os
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.read_heka.i_inj_functions import get_i_inj_step
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cfischer/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell_rounded.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    v_init = -70
    tstop = 500  #5000
    dt = 0.00005
    start_step = 100  #200
    end_step = 400  #4800
    step_amps = [0.8]  #np.arange(0.05, 1.05, 0.05)

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    #cell.soma(.5).nat.ina = 0.0
    #cell.soma(.5).nat.gbar = 0.0
    #cell.soma(.5).nap.gbar = 0.0
    #cell.soma(.5).kdr.gbar = 0.0
    #cell.soma(.5).hcn_slow.gbar = 0.0

    i_channel = cell.soma.record_from('nat', 'ina')

    # fI-curve for model
    v_mat_model = list()
    for step_amp in step_amps:
        i_inj = get_i_inj_step(start_step, end_step, step_amp, tstop, dt)
        simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': v_init, 'tstop': tstop,
                             'dt': dt, 'celsius': 35, 'onset': 0}
        v_model, t_model, _ = iclamp_handling_onset(cell, **simulation_params)
        v_mat_model.append(v_model)

    # plot
    #save_dir_img = os.path.join(save_dir, 'img', 'plot_IV', 'long_steps')
    #if not os.path.exists(save_dir_img):
    #    os.makedirs(save_dir_img)

    np.save(os.path.join('/home/cfischer', 'v_dap.npy'), v_mat_model[0])
    np.save(os.path.join('/home/cfischer', 'i.npy'), i_channel)
    pl.figure()
    pl.plot(t_model, v_mat_model[0], 'r')
    pl.show()

    # # plot all under another with subfigures
    # fig, ax=pl.subplots(20, 1, sharex=True, figsize=(21, 29.7))
    # for i, (amp, v_trace_model) in enumerate(zip(step_amps, v_mat_model)):
    #     ax[i].plot(t_model, v_trace_model, 'r', label='$i_{amp}: $ %.2f' % amp)
    #     ax[i].set_ylim(-80, 60)
    #     ax[i].set_xlim(100, 4900)
    #     ax[i].legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.text(0.06, 0.5, 'Membrane Potential (mV)', va='center', rotation='vertical', fontsize=14)
    # fig.text(0.5, 0.06, 'Time (ms)', ha='center', fontsize=14)
    # #pl.savefig(os.path.join(save_dir_img, 'IV_subplots.pdf'))
    # pl.show()
