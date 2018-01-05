import os
import numpy as np
import matplotlib.pyplot as pl
from cell_fitting.optimization.evaluation.plot_double_ramp.doubleramp_current_threshold import \
    simulate_and_get_current_threshold, plot_current_threshold


def evaluate_double_ramp(pdf, cell, save_dir):
    save_dir_img = os.path.join(save_dir, 'img', 'PP', '125')
    step_amps = [-0.1, 0, 0.1]

    # simulate / load
    current_thresholds = [0] * len(step_amps)
    for i, step_amp in enumerate(step_amps):
        current_thresholds[i], ramp3_times, ramp3_amps, v_dap, t_dap, v_mat, t = simulate_and_get_current_threshold(cell,
                                                                                                            step_amp)
        if step_amp == 0:
            v_mat_step0 = v_mat
    current_threshold_rampIV = float(np.loadtxt(os.path.join(save_dir, 'img', 'rampIV', 'current_threshold.txt')))

    # plot in pdf
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    fig = plot_current_threshold(current_thresholds, current_threshold_rampIV, ramp3_times, step_amps,
                                 ramp3_amps[0], ramp3_amps[-1], v_dap, t_dap, save_dir_img)
    pdf.savefig(fig)
    pl.close()

    fig = plot_double_ramp(current_thresholds, ramp3_amps, t, v_mat_step0, save_dir_img)
    pdf.savefig(fig)
    pl.close()


def plot_double_ramp(current_thresholds, ramp3_amps, t, v_mat_step0, save_dir_img=None):
    amp_idx = np.where(ramp3_amps == np.nanmin(current_thresholds[1]))[0][0]  # step = 0, at minimal current threshold
    fig = pl.figure()
    for v in v_mat_step0[amp_idx, :, :]:
        pl.plot(t, v, 'r')
    pl.xlim(360, 410)
    pl.tight_layout()
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'v.png'))
    return fig
