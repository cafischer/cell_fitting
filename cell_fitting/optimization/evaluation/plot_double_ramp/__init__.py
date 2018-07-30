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


def plot_current_threshold_on_ax(ax, current_thresholds, current_threshold_rampIV, ramp3_times, step_amps,
                                 ramp3_amps, v_dap, t_dap, legend_loc='upper left'):
    colors_dict = {-0.1: 'b', 0.0: 'k', 0.1: 'r'}
    colors = [colors_dict[amp] for amp in step_amps]

    # plot current threshold
    ax2 = ax.twinx()
    ax2.plot(t_dap, v_dap, 'k')
    ax2.set_ylabel('Mem. pot. (mV)', fontsize=12)
    ax2.spines['right'].set_visible(True)

    ax.axhline(ramp3_amps[0], linestyle='--', c='0.5')
    ax.axhline(ramp3_amps[-1], linestyle='--', c='0.5')
    ax.plot(0, current_threshold_rampIV, 'ok', markersize=5)
    for i, current_threshold in enumerate(current_thresholds):
        ax.plot(ramp3_times, current_threshold, '-o', color=colors[i], label='Amp.: ' + str(step_amps[i]),
                markersize=7 - 2.0 * i)
    ax.set_xlabel('Start 2nd pulse (ms)', fontsize=12)
    ax.set_ylabel('Current thresh. (nA)', fontsize=12)
    ax.set_xticks(np.insert(ramp3_times, 0, [0]))
    ax.set_xlim(-0.5, ramp3_times[-1] + 2)
    ax.set_ylim(0, 4.2)
    ax.legend(loc=legend_loc, fontsize=10)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax2.yaxis.set_tick_params(labelsize=10)