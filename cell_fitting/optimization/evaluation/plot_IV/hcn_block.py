import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.plot_IV import simulate_IV
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    step_amp = -0.1

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # simulation
    v_before, t_before = simulate_IV(cell, step_amp, v_init=-75)

    # blocking
    cell.soma(.5).hcn_slow.gbar = 0

    # simulation
    v_after, t_after = simulate_IV(cell, step_amp, v_init=-75)

    # plot
    save_img = os.path.join(save_dir, 'img', 'IV_blockHCN')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    pl.figure()
    pl.plot(t_before, v_before, 'r', label='before ZD')
    pl.plot(t_after, v_after, 'r', label='after ZD', alpha=0.5)
    st = np.ceil(pl.ylim()[1] / 5) * 5
    pl.yticks(np.arange(st, st + 7 * -5, -5))
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend(loc='lower right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_img, str(step_amp)+'(nA).png'))
    pl.show()


    pl.figure()
    pl.plot(t_before, v_before, 'r', label='before ZD')
    pl.plot(t_after, v_after, 'r', label='after ZD', alpha=0.5)
    pl.ylim(-87, -72)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend(loc='lower right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_img, str(step_amp)+'(nA)_zoom.png'))
    pl.show()