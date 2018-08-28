import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.plot_sine_stimulus import simulate_sine_stimulus
from grid_cell_stimuli.spike_phase import plot_phase_hist_on_axes
from cell_characteristics import to_idx
pl.style.use('paper_subplots')


# TODO: colors
# TODO: check simulation_params (e.g. dt)
# TODO: check all exp. data are v_shifted
# TODO: check data computation phase means, stds
if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    v_init = -75
    color_model = '0.5'
    amp1 = 0.4
    amp2 = 0.2
    freq1 = 0.1
    freq2 = 5

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'), mechanism_dir)

    fig = pl.figure(figsize=(8, 6))
    outer = gridspec.GridSpec(2, 2)

    # sine: mem. pot.
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.15, height_ratios=[5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)

    s_ = os.path.join(save_dir_data_plots, 'sine_stimulus/traces/rat', '2015_08_20d',  # using different cell here!
                      str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2))
    v_data = np.load(os.path.join(s_, 'v.npy'))
    t_data = np.load(os.path.join(s_, 't.npy'))
    # v_data += - v_data[0] + v_init
    v_model, t_model, i_inj = simulate_sine_stimulus(cell, amp1, amp2, 1./freq1*1000/2., freq2, 500, 500,
                                                     t_data[1]-t_data[0], v_init=-75, celsius=35, onset=200)

    ax0.plot(t_data, v_data, 'k', label='Exp. cell')
    ax0.plot(t_model, v_model, color_model, label='Model')
    ax1.plot(t_model, i_inj, 'k')

    ax0.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Current (nA)')
    ax1.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.15, 0.5)
    ax1.get_yaxis().set_label_coords(-0.15, 0.5)
    ax1.set_yticks([np.round(np.min(i_inj), 1), np.round(np.max(i_inj), 1)])
    ax0.legend()

    # phase hist.
    ax = pl.Subplot(fig, outer[0, 1])
    fig.add_subplot(ax)

    with open(os.path.join(save_dir_model, model, 'img', 'sine_stimulus/traces',
                           str(amp1) + '_' + str(amp2) + '_' + str(freq1) + '_' + str(freq2), 'phase_hist',
                           'sine_dict.json'), 'r') as f:
        sine_dict_model = json.load(f)

    with open(os.path.join(save_dir_data_plots, 'sine_stimulus/traces/rat', '2015_08_20d',  # using different cell here!
                           str(amp1) + '_' + str(amp2) + '_' + str(freq1) + '_' + str(freq2),
                           'spike_phase', 'sine_dict.json'), 'r') as f:
        sine_dict_data = json.load(f)

    plot_phase_hist_on_axes(ax, 0, [sine_dict_model['phases']], plot_mean=True, color_hist=color_model,
                            alpha=0.6, color_lines=color_model)
    plot_phase_hist_on_axes(ax, 0, [sine_dict_data['phases']], plot_mean=True, color_hist='k',
                            alpha=0.3, color_lines='k')

    ax.set_ylabel('Count')
    ax.set_xlabel('Phase (deg.)')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)

    # mem. pot. per period of fast sine
    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)

    dt = t_model[1] - t_model[0]
    onset_dur = 500
    period = to_idx(1./freq2*1000, dt)
    start_period = 0
    period_half = to_idx(period, 2)
    period_fourth = to_idx(period, 4)
    onset_idx = offset_idx = to_idx(onset_dur, dt)
    period_starts = range(len(t_model))[onset_idx - period_fourth:-offset_idx:period]
    period_ends = range(len(t_model))[onset_idx + period_half + period_fourth:-offset_idx:period]
    period_starts = period_starts[:len(period_ends)]

    colors = pl.cm.get_cmap('Greys')(np.linspace(0.2, 1.0, len(period_starts)))
    for i, (s, e) in enumerate(zip(period_starts, period_ends)):
        ax.plot(t_model[:e - s], v_model[s:e] + i * -10.0, c=colors[i], label=i, linewidth=1)
    ax.set_yticks([])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mem. pot. per up phase')
    ax.set_xlim(0, 200)
    ax.set_ylim(-325, -45)
    ax.get_yaxis().set_label_coords(-0.15, 0.5)

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

    ax.plot(phase_means_data, phase_stds_data, 'o', color='k', alpha=0.5)
    ax.plot(sine_dict_model['mean_phase'], sine_dict_model['std_phase'], 'o', color=color_model, alpha=0.5)

    ax.set_xlabel('Mean phase')
    ax.set_ylabel('Std. phase')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'reproduction_sine.png'))
    pl.show()