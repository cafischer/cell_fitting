import numpy as np
import os
import json
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from nrn_wrapper import Cell, load_mechanism_dir
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
    models = ['2', '3', '4', '5', '6']
    exp_cell = '2015_08_26b'
    color_exp = '#0099cc'
    color_model = 'k'
    amp1s = (0.4, 0.4, 0.7, 0.4, 0.7)
    amp2s = (0.4, 0.4, 0.4, 0.3, 0.5)
    amp1_data = 0.4
    amp2_data = 0.2
    freq1 = 0.1
    freq2 = 5
    standard_sim_params = get_standard_simulation_params()
    load_mechanism_dir(mechanism_dir)

    # plot
    fig = pl.figure(figsize=(12, 9))
    outer = gridspec.GridSpec(3, 5)

    for model_idx, model in enumerate(models):

        # sine: mem. pot.
        inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, model_idx], hspace=0.15,
                                                 height_ratios=[5, 5, 1])
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

        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))  # TODO: cell_rounded
        v_model, t_model, i_inj_model = simulate_sine_stimulus(cell, amp1s[model_idx], amp2s[model_idx],
                                                               1./freq1*1000/2., freq2, 500, 500,
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
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylim(-0.5, 1.3)
        if model_idx == 0:
            ax0.set_ylabel('Mem. pot. (mV)')
            ax2.set_ylabel('Current (nA)')
            ax0.get_yaxis().set_label_coords(-0.25, 0.2)
            ax2.get_yaxis().set_label_coords(-0.25, 0.9)
            ax0.legend()
            ax1.legend()
            ax0.text(-0.4, 1.0, 'A', transform=ax0.transAxes, size=18, weight='bold')

        # phase hist.
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, model_idx], hspace=0.15)
        ax0 = pl.Subplot(fig, inner[0])
        ax1 = pl.Subplot(fig, inner[1])
        fig.add_subplot(ax0)
        fig.add_subplot(ax1)

        with open(os.path.join(save_dir_model, model, 'img', 'sine_stimulus/traces',
                               str(amp1s[model_idx]) + '_' + str(amp2s[model_idx]) + '_' + str(freq1) + '_' + str(freq2),
                               'phase_hist', 'sine_dict.json'), 'r') as f:
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
        ax1.set_xlabel('Phase (deg.)')
        ax0.set_xticks([])
        if model_idx == 0:
            ax0.set_ylabel('Count')
            ax1.set_ylabel('Count')
            ax0.get_yaxis().set_label_coords(-0.25, 0.5)
            ax1.get_yaxis().set_label_coords(-0.25, 0.5)
            ax0.text(-0.4, 1.0, 'B', transform=ax0.transAxes, size=18, weight='bold')

        # time vs phase
        ax = pl.Subplot(fig, outer[2, model_idx])
        fig.add_subplot(ax)
        ax.plot(sine_dict_model['t_phases'], sine_dict_model['phases'], marker='o', color=color_model, linestyle='',
                alpha=0.5)
        ax.plot(sine_dict_data['t_phases'], sine_dict_data['phases'], marker='o', color=color_exp, linestyle='', alpha=0.8)
        slope_model, intercept_model, _, _, _ = linregress(sine_dict_model['t_phases'], sine_dict_model['phases'])
        slope_data, intercept_data, _, _, _ = linregress(sine_dict_data['t_phases'], sine_dict_data['phases'])
        ax.plot(t_model, slope_model * t_model + intercept_model, color_model)
        ax.plot(t_data, slope_data * t_data + 150, color_exp)

        ax.set_ylim(0, 360)
        ax.set_xlabel('Time (ms)')
        if model_idx == 0:
            ax.set_ylabel('Phase (deg.)')
            ax.get_yaxis().set_label_coords(-0.25, 0.5)
            ax.text(-0.4, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')
        print 'slope model: ', sine_dict_model['slope']
        print 'slope data: ', sine_dict_data['slope']

    pl.tight_layout()
    pl.subplots_adjust(bottom=0.06, top=0.97, right=0.99, left=0.06)
    pl.savefig(os.path.join(save_dir_img, 'reproduction_sine_models.png'))
    pl.show()