import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.optimization.evaluation.plot_rampIV import simulate_rampIV, find_current_threshold, plot_rampIV, \
    load_rampIV_data, get_rmse
from nrn_wrapper import Cell, load_mechanism_dir
import time
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'
    model_ids = range(1, 7)
    ramp_amp = 3.1
    mechanism_dir = '../../../model/channels/vavoulis'
    load_mechanism_dir(mechanism_dir)

    v_models = []
    for model_id in model_ids:
        model_dir = os.path.join(save_dir, str(model_id), 'cell.json')
        cell = Cell.from_modeldir(model_dir)

        v, t, _ = simulate_rampIV(cell, ramp_amp, v_init=-75)
        v_models.append(v)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'rampIV')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    n_rows = 2
    n_columns = 3
    fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, 8.5))
    cell_idx = 0
    for i1 in range(n_rows):
        for i2 in range(n_columns):
            if cell_idx < len(model_ids):
                #axes[i1, i2].set_title(model_ids[cell_idx], fontsize=12)
                axes[i1, i2].plot(t, v_models[cell_idx], 'k')
                if i1 == (n_rows - 1):
                    axes[i1, i2].set_xlabel('Time (ms)')
                if i2 == 0:
                    axes[i1, i2].set_ylabel('Membrane Potential (mV)')
            else:
                axes[i1, i2].spines['left'].set_visible(False)
                axes[i1, i2].spines['bottom'].set_visible(False)
                axes[i1, i2].set_xticks([])
                axes[i1, i2].set_yticks([])
            cell_idx += 1
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'v_at_%.2f(nA).png' % ramp_amp))
    pl.show()