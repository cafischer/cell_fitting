from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.evaluation.plot_double_ramp.doubleramp_current_threshold import simulate_and_get_current_threshold
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    model_ids = range(1, 7)
    AP_threshold = 0
    step_amps = [-0.1, 0, 0.1]
    save_dir_img = os.path.join(save_dir, 'img', 'PP', '125')
    mechanism_dir = '../../../model/channels/vavoulis'
    load_mechanism_dir(mechanism_dir)
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    v_dap_models = []
    current_threshold_rampIV_models = []
    current_thresholds_models = []
    for model_id in model_ids:
        print model_id
        # load model
        model_dir = os.path.join(save_dir, str(model_id), 'cell.json')
        cell = Cell.from_modeldir(model_dir)

        # load current threshold
        current_threshold_rampIV = float(np.loadtxt(os.path.join(save_dir, str(model_id), 'img', 'rampIV',
                                                                 'current_threshold.txt')))

        # simulation
        current_thresholds = [0] * len(step_amps)
        for i, step_amp in enumerate(step_amps):
            current_thresholds[i], ramp3_times, ramp3_amps, v_dap, t_dap, v_mat, t = simulate_and_get_current_threshold(cell, step_amp)
            if i == 1:
                v_dap_models.append(v_dap)
        current_threshold_rampIV_models.append(current_threshold_rampIV)
        current_thresholds_models.append(current_thresholds)


    # plot current threshold
    colors_dict = {-0.1: 'b', 0.0: 'k', 0.1: 'r'}
    colors = [colors_dict[amp] for amp in step_amps]
    n_rows = 2
    n_columns = 3
    fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, 8.5))
    cell_idx = 0
    for i1 in range(n_rows):
        for i2 in range(n_columns):
            if cell_idx < len(model_ids):
                # axes[i1, i2].set_title(model_ids[cell_idx], fontsize=12)
                ax2 = axes[i1, i2].twinx()
                ax2.plot(t_dap, v_dap_models[cell_idx], 'k')
                ax2.spines['right'].set_visible(True)
                ax2.set_yticks(np.arange(-80, 60, 20))
                ax2.set_ylim(-80, 60)
                ax2.set_yticklabels([])

                # axes[i1, i2].axhline(ramp3_amp_min, linestyle='--', c='0.5')
                # axes[i1, i2].axhline(ramp3_amp_max, linestyle='--', c='0.5')
                axes[i1, i2].plot(0, current_threshold_rampIV_models[cell_idx], 'ok', markersize=6.5)
                for i, current_threshold in enumerate(current_thresholds_models[cell_idx]):
                    axes[i1, i2].plot(ramp3_times, current_threshold, '-o', color=colors[i],
                                      label='Step Amp.: ' + str(step_amps[i]), markersize=9 - 2.5 * i)
                axes[i1, i2].set_xticks(np.insert(ramp3_times, 0, [0]))
                axes[i1, i2].set_xlim(-0.5, ramp3_times[-1] + 2)
                axes[i1, i2].set_yticks(np.arange(0, 3.5, 0.5))
                axes[i1, i2].set_ylim(0, 3.2)
                axes[i1, i2].legend(loc='upper right')

                if i1 == (n_rows - 1):
                    axes[i1, i2].set_xlabel('Time (ms)')
                if i2 == 0:
                    axes[i1, i2].set_ylabel('Current threshold (nA)')
                elif i2 == (n_columns - 1):
                    ax2.set_ylabel('Membrane potential (mV)')
                    ax2.set_yticklabels(np.arange(-80, 60, 20))

            else:
                axes[i1, i2].spines['left'].set_visible(False)
                axes[i1, i2].spines['bottom'].set_visible(False)
                axes[i1, i2].set_xticks([])
                axes[i1, i2].set_yticks([])
            cell_idx += 1
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'current_threshold.png'))
    pl.show()