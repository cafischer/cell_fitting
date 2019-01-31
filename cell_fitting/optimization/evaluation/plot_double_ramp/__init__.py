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
                                 ramp3_amps, v_dap, t_dap, legend_loc='upper left', colors_dict=None, v_init=-75,
                                 label=True, plot_range=True, with_right_spine=False, shift_to_rest=False):
    if colors_dict is None:
        colors_dict = {-0.1: 'k', 0.0: 'k', 0.1: 'k'}
    marker_dict = {-0.1: 'v', 0.0: 'o', 0.1: '^'}
    colors = [colors_dict[amp] for amp in step_amps]
    markers = [marker_dict[amp] for amp in step_amps]
    if label is True:
        labels = ['Amp.: ' + str(amp) for amp in step_amps]
    else:
        labels = ['' for amp in step_amps]

    # plot current threshold
    ax2 = ax.twinx()
    if shift_to_rest:
        v_dap = np.array(v_dap) - v_dap[0]
        ax2.set_ylim(0, 120)
    else:
        v_dap = np.array(v_dap) - v_dap[0] + v_init
        ax2.set_ylim(-80, 20)
    ax2.plot(t_dap, v_dap, color=colors_dict[0.0], linestyle=':')
    if with_right_spine:
        ax2.spines['right'].set_visible(True)
        ax2.set_ylabel('Mem. pot. (mV)')
    else:
        ax2.set_yticks([])

    if plot_range:
        ax.axhline(ramp3_amps[0], linestyle='--', c='0.5')
        ax.axhline(ramp3_amps[-1], linestyle='--', c='0.5')
    ax.plot(0, current_threshold_rampIV, 'o', color=colors_dict[0.0], clip_on=False)
    ramp3_peak_times = np.array(ramp3_times) + 1.0
    for i, current_threshold in enumerate(current_thresholds):
        ax.plot(ramp3_peak_times, current_threshold, linestyle='-', marker=markers[i], color=colors[i],
                label=labels[i], clip_on=False)
    ax.set_xlabel('Midpoint 2nd pulse (ms)')
    #ax.set_ylabel('Current thresh. (nA)')
    ax.set_xticks(np.insert(ramp3_peak_times, 0, [0]))
    ax.set_xlim(-0.5, ramp3_peak_times[-1] + 2)
    ax.set_ylim(0, 3.5)
    if legend_loc is not None:
        ax.legend(loc=legend_loc)