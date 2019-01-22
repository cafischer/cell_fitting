import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.optimization.evaluation.plot_sine_stimulus import simulate_sine_stimulus, get_sine_stimulus
from grid_cell_stimuli.spike_phase import plot_phase_hist_on_axes
from cell_characteristics import to_idx
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    color_exp = '#0099cc'
    color_model = 'k'
    amp1 = 0.4
    amp2 = 0.4
    amp1_data = 0.4
    amp2_data = 0.2
    freq1 = 0.1
    freq2 = 5
    standard_sim_params = get_standard_simulation_params()

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'), mechanism_dir)

    fig = pl.figure(figsize=(8, 6))
    outer = gridspec.GridSpec(2, 2)

    # sine: mem. pot.
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 0], hspace=0.15, height_ratios=[5, 5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    ax2 = pl.Subplot(fig, inner[2])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)

    s_ = os.path.join(save_dir_data_plots, 'sine_stimulus/traces/rat', '2015_08_20d',  # using different cell here!
                      str(amp1_data)+'_'+str(amp2_data)+'_'+str(freq1)+'_'+str(freq2))
    v_data = np.load(os.path.join(s_, 'v.npy'))
    t_data = np.load(os.path.join(s_, 't.npy'))
    dt_data = t_data[1]-t_data[0]
    i_inj_data = get_sine_stimulus(amp1_data, amp2_data, 1./freq1*1000/2., freq2, 500, 500-dt_data, dt_data)
    v_model, t_model, i_inj_model = simulate_sine_stimulus(cell, amp1, amp2, 1./freq1*1000/2., freq2, 500, 500,
                                                           **standard_sim_params)

    start_i_inj_data = np.where(i_inj_data)[0][0]
    start_i_inj_model = np.where(i_inj_model)[0][0]
    vrest_data = np.mean(v_data[:start_i_inj_data])
    vrest_model = np.mean(v_model[:start_i_inj_model])
    # ax0.plot(t_data, v_data, color_exp, linewidth=0.5, label='Data')
    # ax1.plot(t_model, v_model, color_model, linewidth=0.5, label='Model')
    # ax0.set_ylim(-100, 50)
    # ax1.set_ylim(-100, 50)
    ax0.plot(t_data, v_data - vrest_data, color_exp, linewidth=0.5, label='Data')
    ax1.plot(t_model, v_model - vrest_model, color_model, linewidth=0.5, label='Model')
    ax2.plot(t_data, i_inj_data, color_exp)
    ax2.plot(t_model, i_inj_model, color_model)
    ax0.set_ylim(-25, 135)
    ax1.set_ylim(-25, 135)

    ax0.set_xticks([])
    ax1.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax2.set_ylabel('Current (nA)')
    ax2.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.15, 0.2)
    ax2.get_yaxis().set_label_coords(-0.15, 0.9)
    ax2.set_yticks([np.round(np.min(i_inj_model), 1), np.round(np.max(i_inj_model), 1)])
    ax0 .legend()
    ax1.legend()
    ax0.text(-0.25, 1.0, 'A', transform=ax0.transAxes, size=18, weight='bold')

    # phase hist.
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], hspace=0.15)
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    with open(os.path.join(save_dir_model, model, 'img', 'sine_stimulus/traces',
                           str(amp1) + '_' + str(amp2) + '_' + str(freq1) + '_' + str(freq2), 'phase_hist',
                           'sine_dict.json'), 'r') as f:
        sine_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'sine_stimulus/traces/rat', '2015_08_20d',  # using different cell here!
                           str(amp1_data) + '_' + str(amp2_data) + '_' + str(freq1) + '_' + str(freq2),
                           'spike_phase', 'sine_dict.json'), 'r') as f:
        sine_dict_data = json.load(f)

    plot_phase_hist_on_axes(ax0, 0, [sine_dict_data['phases']], plot_mean=True, color_hist=color_exp,
                            alpha=0.5, color_lines=color_exp)
    plot_phase_hist_on_axes(ax1, 0, [sine_dict_model['phases']], plot_mean=True, color_hist=color_model,
                            alpha=0.5, color_lines=color_model)

    ax0.set_ylim(0, 11)
    ax1.set_ylim(0, 11)
    ax0.set_ylabel('Frequency')
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Phase (deg.)')
    ax0.set_xticks([])
    ax0.get_yaxis().set_label_coords(-0.1, 0.5)
    ax1.get_yaxis().set_label_coords(-0.1, 0.5)
    ax0.text(-0.25, 1.0, 'B', transform=ax0.transAxes, size=18, weight='bold')

    # # mem. pot. per period of fast sine
    # ax = pl.Subplot(fig, outer[1, 0])
    # fig.add_subplot(ax)
    #
    # dt = t_model[1] - t_model[0]
    # onset_dur = 500
    # period = to_idx(1./freq2*1000, dt)
    # start_period = 0
    # period_half = to_idx(period, 2)
    # period_fourth = to_idx(period, 4)
    # onset_idx = offset_idx = to_idx(onset_dur, dt)
    # period_starts = range(len(t_model))[onset_idx - period_fourth:-offset_idx:period]
    # period_ends = range(len(t_model))[onset_idx + period_half + period_fourth:-offset_idx:period]
    # period_starts = period_starts[:len(period_ends)]
    #
    # colors = pl.cm.get_cmap('Greys')(np.linspace(0.2, 1.0, len(period_starts)))
    # for i, (s, e) in enumerate(zip(period_starts, period_ends)):
    #     ax.plot(t_model[:e - s], v_model[s:e] + i * -10.0, c=colors[i], label=i, linewidth=1)
    # ax.set_yticks([])
    # ax.set_xlabel('Time (ms)')
    # ax.set_ylabel('Mem. pot. per up phase')
    # ax.set_xlim(0, 200)
    # ax.set_ylim(-325, -45)
    # ax.get_yaxis().set_label_coords(-0.15, 0.5)
    # ax.text(-0.25, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')

    # time vs phase
    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)
    ax.plot(sine_dict_model['t_phases'], sine_dict_model['phases'], marker='o', color=color_model, linestyle='',
            alpha=0.5)
    ax.plot(sine_dict_data['t_phases'], sine_dict_data['phases'], marker='o', color=color_exp, linestyle='', alpha=0.8)
    slope_model, intercept_model, _, _, _ = linregress(sine_dict_model['t_phases'], sine_dict_model['phases'])
    slope_data, intercept_data, _, _, _ = linregress(sine_dict_data['t_phases'], sine_dict_data['phases'])
    ax.plot(t_model, slope_model * t_model + intercept_model, color_model)
    ax.plot(t_data, slope_data * t_data + 150, color_exp)
    ax.set_ylim(0, 360)
    ax.set_ylabel('Phase (deg.)')
    ax.set_xlabel('Time (ms)')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.25, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')
    print 'slope model: ', sine_dict_model['slope']
    print 'slope data: ', sine_dict_data['slope']

    # mean vs std phase for all cells
    ax = pl.Subplot(fig, outer[1, 1])
    fig.add_subplot(ax)

    amp1_data = None
    amp2_data = None
    phase_means_data = np.load(os.path.join(save_dir_data_plots, 'sine_stimulus', 'traces', 'rat', 'summary',
                                            'spike_phase',
                                            str(amp1_data)+'_'+str(amp2_data)+'_'+str(freq1)+'_'+str(freq2),
                                            'phase_means.npy'))
    phase_stds_data = np.load(os.path.join(save_dir_data_plots, 'sine_stimulus', 'traces', 'rat', 'summary',
                                            'spike_phase',
                                            str(amp1_data)+'_'+str(amp2_data)+'_'+str(freq1)+'_'+str(freq2),
                                            'phase_stds.npy'))

    ax.plot(phase_means_data, phase_stds_data, 'o', color=color_exp, alpha=0.5)
    ax.plot(sine_dict_model['mean_phase'], sine_dict_model['std_phase'], 'o', color=color_model, alpha=0.5)

    ax.set_xlabel('Mean phase (deg.)')
    ax.set_ylabel('Std. phase (deg.)')
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    ax.text(-0.25, 1.0, 'D', transform=ax.transAxes, size=18, weight='bold')

    pl.tight_layout()
    pl.subplots_adjust(left=0.1, top=0.96, bottom=0.08)
    pl.savefig(os.path.join(save_dir_img, 'reproduction_sine.png'))
    pl.show()