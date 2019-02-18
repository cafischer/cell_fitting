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
from cell_fitting.util import init_nan
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
pl.style.use('paper_subplots')


def get_phase_first_spike_in_period(phases, t_phases, v, t, dt):
    onset_dur = 500
    period_len_idx = to_idx(1. / freq2 * 1000, dt)
    period_half = to_idx(period_len_idx, 2)
    period_fourth = to_idx(period_len_idx, 4)
    onset_idx = offset_idx = to_idx(onset_dur, dt)
    period_start_idxs = range(len(t))[onset_idx - period_fourth:-offset_idx:period_len_idx]
    period_end_idxs = range(len(t))[onset_idx + period_half + period_fourth:-offset_idx:period_len_idx]
    period_start_idxs = period_start_idxs[:len(period_end_idxs)]
    t_phase_idxs = np.array([to_idx(tp, dt, 3) for tp in t_phases])
    # pl.figure()
    # pl.plot(t_data, v_data, 'k')
    # pl.plot(t_data[period_start_idxs], v_data[period_start_idxs], 'or')
    # pl.plot(t_data[period_end_idxs], v_data[period_end_idxs], 'ob')
    # pl.show()
    first_phases = init_nan(len(period_start_idxs))
    for period in range(len(period_start_idxs)):
        phases_period = phases[np.logical_and(period_start_idxs[period] < t_phase_idxs, t_phase_idxs < period_end_idxs[period])]
        if len(phases_period >= 1):
            first_phases[period] = phases_period[0]
        # pl.figure()
        # pl.plot(t[period_start_idxs[period]: period_end_idxs[period]], v[period_start_idxs[period]: period_end_idxs[period]], 'k')
        # pl.show()
    return first_phases


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
    ax0.plot(t_data/1000., v_data - vrest_data, color_exp, linewidth=0.5, label='Data')
    ax1.plot(t_model/1000., v_model - vrest_model, color_model, linewidth=0.5, label='Model')
    ax2.plot(t_data/1000., i_inj_data, color_exp)
    ax2.plot(t_model/1000., i_inj_model, color_model)
    ax0.set_ylim(-25, 135)
    ax1.set_ylim(-25, 135)
    ax0.set_xlim(0, t_data[-1]/1000.)
    ax1.set_xlim(0, t_data[-1]/1000.)
    ax2.set_xlim(0, t_data[-1] / 1000.)
    ax0.set_xticks([])
    ax1.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax2.set_ylabel('Current (nA)')
    ax2.set_xlabel('Time (s)')
    ax0.get_yaxis().set_label_coords(-0.15, 0.2)
    ax2.get_yaxis().set_label_coords(-0.15, 0.9)
    ax2.set_yticks([np.round(np.min(i_inj_model), 1), np.round(np.max(i_inj_model), 1)])
    ax0.legend()
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
    ax1.set_xticks([0, 90, 180, 270, 360])
    ax1.set_xticklabels([0, 90, 180, 270, 360])
    ax0.get_yaxis().set_label_coords(-0.12, 0.5)
    ax1.get_yaxis().set_label_coords(-0.12, 0.5)
    ax0.text(-0.25, 1.0, 'B', transform=ax0.transAxes, size=18, weight='bold')

    # get variation of the phase of the first spike with respect to the theta oscillation
    first_phases_data = get_phase_first_spike_in_period(np.array(sine_dict_data['phases']),
                                                        np.array(sine_dict_data['t_phases']),
                                                        v_data, t_data, dt_data)
    first_phases_model = get_phase_first_spike_in_period(np.array(sine_dict_model['phases']),
                                                         np.array(sine_dict_model['t_phases']),
                                                         v_model, t_model, t_model[1]-t_model[0])
    print 'Phase first spike per period (data): ', np.nanmean(first_phases_data), np.nanstd(first_phases_data)
    print 'Phase first spike per period (model): ', np.nanmean(first_phases_model), np.nanstd(first_phases_model)

    # time vs phase
    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)
    ax.plot(np.array(sine_dict_model['t_phases']) / 1000., sine_dict_model['phases'], marker='o', color=color_model, linestyle='',
            alpha=0.5)
    ax.plot(np.array(sine_dict_data['t_phases']) / 1000., sine_dict_data['phases'], marker='o', color=color_exp, linestyle='', alpha=0.8)
    slope_model, intercept_model, _, _, _ = linregress(sine_dict_model['t_phases'], sine_dict_model['phases'])
    slope_data, intercept_data, _, _, _ = linregress(sine_dict_data['t_phases'], sine_dict_data['phases'])
    ax.plot(t_model / 1000., slope_model * t_model + intercept_model, color_model)
    ax.plot(t_data / 1000., slope_data * t_data + 150, color_exp)
    ax.set_ylim(0, 360)
    ax.set_xlim(0, t_data[-1] / 1000.)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_ylabel('Phase (deg.)')
    ax.set_xlabel('Time (s)')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.25, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')
    print 'slope model (deg./s): ', sine_dict_model['slope'] *  1000.
    print 'slope data (deg./s): ', sine_dict_data['slope'] * 1000.

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

    axins = inset_axes(ax, width='65%', height='50%', loc=1)
    axins.plot(phase_means_data, phase_stds_data, 'o', color=color_exp, alpha=0.5)
    axins.plot(sine_dict_model['mean_phase'], sine_dict_model['std_phase'], 'o', color=color_model, alpha=0.5)
    axins.set_xlim(122, 184)
    axins.set_ylim(18, 95)
    #axins.set_xticks(np.arange(120, 200, 20))
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_xlabel('Mean phase (deg.)')
    ax.set_ylabel('Std. phase (deg.)')
    ax.get_yaxis().set_label_coords(-0.12, 0.5)
    ax.text(-0.25, 1.0, 'D', transform=ax.transAxes, size=18, weight='bold')

    pl.tight_layout()
    pl.subplots_adjust(left=0.1, top=0.96, bottom=0.08)
    pl.savefig(os.path.join(save_dir_img, 'reproduction_sine.png'))
    pl.show()