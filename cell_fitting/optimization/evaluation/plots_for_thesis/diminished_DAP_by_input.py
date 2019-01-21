import numpy as np
import os
import matplotlib.pyplot as pl
from cell_fitting.read_heka import load_data
from cell_characteristics import to_idx
from cell_fitting.util import change_color_brightness
from matplotlib.colors import to_rgb
import matplotlib.gridspec as gridspec
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    exp_cell = '2015_08_26b'
    color_exp = '#0099cc'
    ramp_amp = 3.5

    # load data
    v_exp, t_exp, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'rampIV', ramp_amp)
    v_exp_step, t_exp_step, i_inj_step = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'IV', 0.3)
    dt = t_exp[1] - t_exp[0]
    dt_step = t_exp_step[1] - t_exp_step[0]

    # plot
    fig = pl.figure(figsize=(6, 4.5))
    outer = gridspec.GridSpec(1, 1)
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.1, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    ax0.plot(t_exp[to_idx(0, dt):to_idx(60, dt)], v_exp[to_idx(0, dt):to_idx(60, dt)],
             color=change_color_brightness(to_rgb(color_exp), 35, 'brighter'), label='Triangular pulse')

    v_diff = v_exp[to_idx(0, dt)] - v_exp_step[to_idx(660, dt_step)]
    ax0.plot(t_exp_step[to_idx(659, dt_step):to_idx(719, dt_step)] - t_exp_step[to_idx(659, dt_step)],
             v_exp_step[to_idx(659, dt_step):to_idx(719, dt_step)] + v_diff,
             color=change_color_brightness(to_rgb(color_exp), 35, 'darker'), label='Step current')

    ax1.plot(t_exp[to_idx(0, dt):to_idx(60, dt)], i_inj[to_idx(0, dt):to_idx(60, dt)],
             color=change_color_brightness(to_rgb(color_exp), 35, 'brighter'))
    ax1.plot(t_exp_step[to_idx(659, dt_step):to_idx(719, dt_step)] - t_exp_step[to_idx(659, dt_step)],
             i_inj_step[to_idx(659, dt_step):to_idx(719, dt_step)],
             color=change_color_brightness(to_rgb(color_exp), 35, 'darker'))

    ax0.legend()
    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.1, 0.5)
    ax1.get_yaxis().set_label_coords(-0.1, 0.5)
    ax1.set_yticks([np.min(i_inj), np.max(i_inj)])

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'diminished_DAP_by_input.png'))
    pl.show()


    # quantify difference in DAP
    DAP_amp_ramp, DAP_deflection_ramp, DAP_max_idx_ramp = get_spike_characteristics(v_exp, t_exp, ['DAP_amp', 'DAP_deflection', 'DAP_max_idx'], v_exp[0],
                                                                  check=False, **get_spike_characteristics_dict())
    DAP_amp_step, DAP_deflection_step, DAP_max_idx_step = get_spike_characteristics(v_exp_step[to_idx(659, dt_step):to_idx(719, dt_step)],
                                             t_exp_step[to_idx(659, dt_step):to_idx(719, dt_step)] - t_exp_step[to_idx(659, dt_step)],
                                             ['DAP_amp', 'DAP_deflection', 'DAP_max_idx'], v_exp_step[to_idx(659, dt_step)], check=False,
                                             **get_spike_characteristics_dict())

    print v_exp[DAP_max_idx_ramp]
    print v_exp_step[to_idx(659, dt_step):to_idx(719, dt_step)][DAP_max_idx_step] + v_diff
    print ''
