import numpy as np
import os
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import json
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.read_heka import get_i_inj_standard_params
from cell_fitting.optimization.evaluation import simulate_model, get_spike_characteristics_dict
from cell_fitting.optimization.evaluation.plot_rampIV import simulate_rampIV
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.util import characteristics_dict_for_plotting
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import compute_v_sag_and_steady_state
from cell_fitting.optimization.evaluation.plot_IV import simulate_and_compute_fI_curve, \
    fit_fI_curve
from cell_fitting.optimization.evaluation.plot_IV.latency_vs_ISI12_distribution import get_latency_and_ISI12
from cell_fitting.optimization.evaluation.plot_zap import simulate_and_compute_zap_characteristics
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    models = ['2', '3', '4', '5', '6']
    exp_cell = '2015_08_26b'
    exp_cell_dr = '2015_08_06d'
    color_exp = '#0099cc'
    color_model = 'k'
    standard_sim_params = get_standard_simulation_params()
    load_mechanism_dir(mechanism_dir)

    # plot
    fig = pl.figure(figsize=(8, 10))
    outer = gridspec.GridSpec(4, 2)

    # distribution of DAP characteristics
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 0], wspace=0.5, hspace=1.2)
    axes = [inner[0, 0], inner[0, 1], inner[1, 0], inner[1, 1]]
    characteristics = ['DAP_deflection', 'DAP_amp', 'DAP_time', 'DAP_width']
    units = ['mV', 'mV', 'ms', 'ms']
    characteristics_dict_plot = characteristics_dict_for_plotting()

    characteristics_mat_exp = np.load(os.path.join(save_dir_data_plots, 'spike_characteristics/rat',
                                                   'characteristics_mat.npy')).astype(float)
    characteristics_exp = np.load(os.path.join(save_dir_data_plots, 'spike_characteristics/rat',
                                               'return_characteristics.npy'))

    for characteristic_idx, characteristic in enumerate(characteristics):
        ax = pl.Subplot(fig, axes[characteristic_idx])
        fig.add_subplot(ax)

        characteristic_idx_exp = np.where(characteristic == characteristics_exp)[0][0]
        not_nan_exp = ~np.isnan(characteristics_mat_exp[:, characteristic_idx_exp])
        ax.hist(characteristics_mat_exp[:, characteristic_idx_exp][not_nan_exp], bins=100, color=color_exp,
                label='Data')

        characteristics_each_model = np.zeros(len(models))
        for model_idx, model in enumerate(models):
            cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'))
            v, t, i_inj = simulate_rampIV(cell, 3.5, v_init=-75)
            start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
            v_rest = np.mean(v[0:start_i_inj])
            characteristics_mat_model = np.array(get_spike_characteristics(v, t, characteristics, v_rest, check=False,
                                                                           **get_spike_characteristics_dict()),
                                                 dtype=float)
            ax.axvline(characteristics_mat_model[characteristic_idx], 0, 0.95, color=color_model, linewidth=1.0,
                       label='Model')
            characteristics_each_model[model_idx] = characteristics_mat_model[characteristic_idx]

        order_models = np.argsort(characteristics_each_model) + 1
        ax.annotate(str(order_models).replace('[', '').replace(']', ''),
                    xy=(characteristics_mat_model[characteristic_idx], ax.get_ylim()[1]+0.1),
                    color=color_model, fontsize=8, ha='center')

        ax.set_xlabel(characteristics_dict_plot[characteristic] + ' ('+units[characteristic_idx]+')', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.get_yaxis().set_label_coords(-0.25, 0.5)
        ax.get_xaxis().set_label_coords(0.5, -0.4)
        ax.set_xlim(0, None)

        if characteristic_idx == 0:
            ax.text(-0.74, 1.0, 'A', transform=ax.transAxes, size=18, weight='bold')

    # sag
    ax = pl.Subplot(fig, outer[0, 1])
    fig.add_subplot(ax)

    step_amp = -0.1
    sag_deflections_data = np.load(os.path.join(save_dir_data_plots, 'IV', 'sag', 'rat', str(step_amp),
                                         'sag_amps.npy'))
    steady_state_amp = np.load(os.path.join(save_dir_data_plots, 'IV', 'sag', 'rat', str(step_amp),
                                              'v_deflections.npy'))
    ax.plot(sag_deflections_data, steady_state_amp, 'o', color=color_exp, alpha=0.5, label='Data')

    axins = inset_axes(ax, width='50%', height='50%', loc=1)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    axins.plot(sag_deflections_data, steady_state_amp, 'o', color=color_exp, alpha=0.5, label='Data')

    for model_idx, model in enumerate(models):
        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'))
        v_model, t_model, i_inj_model = simulate_model(cell, 'IV', step_amp, 1149.95, **standard_sim_params)
        start_step_idx = np.nonzero(i_inj_model)[0][0]
        end_step_idx = np.nonzero(i_inj_model)[0][-1] + 1
        v_sags, v_steady_states, _ = compute_v_sag_and_steady_state([v_model], [step_amp], AP_threshold=0,
                                                                    start_step_idx=start_step_idx,
                                                                    end_step_idx=end_step_idx)
        sag_deflection_model = v_steady_states[0] - v_sags[0]
        vrest = np.mean(v_model[:start_step_idx])
        steady_state_amp_model = vrest - v_steady_states[0]
        ax.plot(sag_deflection_model, steady_state_amp_model, 'o', color=color_model, alpha=0.5,
                label='Model' if model_idx == 0 else '')
        #ax.annotate(str(model_idx+1), xy=(sag_amp_model+0.005, v_deflection_m  odel+0.05))
        axins.plot(sag_deflection_model, steady_state_amp_model, 'o', color=color_model, alpha=0.5,
                   label='Model' if model_idx == 0 else '')
        axins.annotate(str(model_idx + 1), xy=(sag_deflection_model + 0.01, steady_state_amp_model - 0.15), va='top', fontsize=8, )
    axins.set_ylim(1.1, 4.8)
    axins.set_xlim(0.25, 1.5)
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)

    ax.set_ylim(0, 47.8)
    ax.set_xlim(0, 14.5)
    ax.set_xlabel('Sag deflection (mV)')
    ax.set_ylabel('Steady state amp. (mV)')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.3, 1.0, 'B', transform=ax.transAxes, size=18, weight='bold')

    # latency vs ISI1/2
    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)

    latency_data = np.load(os.path.join(save_dir_data_plots, 'IV/latency_vs_ISI12/rat', 'latency.npy'))
    ISI12_data = np.load(os.path.join(save_dir_data_plots, 'IV/latency_vs_ISI12/rat', 'ISI12.npy'))
    ax.plot(latency_data[latency_data >= 0], ISI12_data[latency_data >= 0], 'o', color=color_exp,
                    alpha=0.5, label='Data', clip_on=False)

    for model_idx, model in enumerate(models):
        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'))
        latency_model, ISI12_model = get_latency_and_ISI12(cell)
        ax.plot(latency_model, ISI12_model, 'o', color=color_model, alpha=0.5)
        if model_idx == 0:
            ax.annotate(str(model_idx + 1), xy=(latency_model + 9, ISI12_model + 0.0), fontsize=8)
        else:
            ax.annotate(str(model_idx + 1), xy=(latency_model + 1, ISI12_model + 0.1), fontsize=8)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel('Latency of the first spike (ms)')
    ax.set_ylabel('$ISI_{1/2}$ (ms)')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.3, 1.0, 'C', transform=ax.transAxes, size=18, weight='bold')

    # fit F-I curve
    ax1 = pl.Subplot(fig, outer[1, 1])
    fig.add_subplot(ax1)
    ax2 = pl.Subplot(fig, outer[2, 0])
    fig.add_subplot(ax2)
    ax3 = pl.Subplot(fig, outer[2, 1])
    fig.add_subplot(ax3)
    FI_a = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'FI_a.npy'))
    FI_b = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'FI_b.npy'))
    FI_c = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'FI_c.npy'))
    RMSE = np.load(os.path.join(save_dir_data_plots, 'IV/fi_curve/rat', 'RMSE.npy'))

    ax1.plot(FI_a, FI_b, 'o', color=color_exp, alpha=0.5)
    ax2.plot(FI_b, FI_c, 'o', color=color_exp, alpha=0.5)
    ax3.plot(FI_c, FI_a, 'o', color=color_exp, alpha=0.5)

    for model_idx, model in enumerate(models):
        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'))
        amps_greater0, firing_rates_model = simulate_and_compute_fI_curve(cell)
        FI_a_model, FI_b_model, FI_c_model, RMSE_model = fit_fI_curve(amps_greater0, firing_rates_model)

        ax1.plot(-1000, -1000, 'o', color=color_exp, alpha=0.5, label='Data' if model_idx == 0 else '')  # fake plot for legend
        ax1.plot([FI_a_model], [FI_b_model], 'o', color=color_model, alpha=0.5, label='Model' if model_idx == 0 else '')
        ax1.annotate(str(model_idx + 1), xy=(FI_a_model + 4, FI_b_model + 0.01), fontsize=8)
        ax1.set_xlim(0, 380)
        ax1.set_xticks(np.arange(0, 380, 50))
        ax1.set_ylim(0, 0.8)
        ax1.set_yticks(np.arange(0, 0.9, 0.2))
        if model_idx == 0:
            ax1.set_xlabel('a')
            ax1.set_ylabel('b')
            ax1.get_yaxis().set_label_coords(-0.15, 0.5)
            ax1.text(-0.3, 1.0, 'D', transform=ax1.transAxes, size=18, weight='bold')
            ax1.legend()


        ax2.plot([FI_b_model], [FI_c_model], 'o', color=color_model, alpha=0.5)
        ax2.annotate(str(model_idx + 1), xy=(FI_b_model + 0.01, FI_c_model + 0.01), fontsize=8)
        ax2.set_xlim(0, 0.8)
        ax2.set_xticks(np.arange(0, 0.9, 0.2))
        ax2.set_ylim(0, 1.0)
        ax2.set_yticks(np.arange(0, 1.1, 0.2))
        if model_idx == 0:
            ax2.set_xlabel('b')
            ax2.set_ylabel('c')
            ax2.get_yaxis().set_label_coords(-0.15, 0.5)
            ax2.text(-0.3, 1.0, 'E', transform=ax2.transAxes, size=18, weight='bold')

        ax3.plot([FI_c_model], [FI_a_model], 'o', color=color_model, alpha=0.5)
        ax3.annotate(str(model_idx + 1), xy=(FI_c_model + 0.016, FI_a_model + 0.8), fontsize=8)
        ax3.set_xlim(0, 1.0)
        ax3.set_xticks(np.arange(0, 1.1, 0.2))
        ax3.set_ylim(0, 380)
        ax3.set_yticks(np.arange(0, 380, 50))
        if model_idx == 0:
            ax3.set_xlabel('c')
            ax3.set_ylabel('a')
            ax3.get_yaxis().set_label_coords(-0.15, 0.5)
            ax3.text(-0.3, 1.0, 'F', transform=ax3.transAxes, size=18, weight='bold')

    # resonance
    ax = pl.Subplot(fig, outer[3, 0])
    fig.add_subplot(ax)

    res_freqs_data = np.load(os.path.join(save_dir_data_plots, 'Zap20/rat/summary', 'res_freqs.npy'))
    q_values_data = np.load(os.path.join(save_dir_data_plots, 'Zap20/rat/summary', 'q_values.npy'))

    ax.plot(res_freqs_data, q_values_data, 'o', color=color_exp, alpha=0.5, label='Data', clip_on=False)

    for model_idx, model in enumerate(models):
        zap_params = get_i_inj_standard_params('Zap20')
        zap_params['tstop'] = 34000 - standard_sim_params['dt']
        zap_params['dt'] = standard_sim_params['dt']
        zap_params['offset_dur'] = zap_params['onset_dur'] - standard_sim_params['dt']
        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'))
        if model == '4':
            zap_params['amp'] = 0.08
        v_model, t_model, i_inj_model, imp_smooth_model, frequencies_model, \
        res_freq_model, q_value_model = simulate_and_compute_zap_characteristics(cell, zap_params)
        ax.plot(res_freq_model, q_value_model, 'o', color=color_model, alpha=0.5, label='Model')
        if model_idx == 0:
            ax.annotate(str(model_idx + 1), xy=(res_freq_model + -0.13, q_value_model + 0.07), fontsize=8)
        else:
            ax.annotate(str(model_idx + 1), xy=(res_freq_model + 0.1, q_value_model + 0.07), fontsize=8)

    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.set_xlabel('Q-value')
    ax.set_ylabel('Res. freq. (Hz)')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.3, 1.0, 'G', transform=ax.transAxes, size=18, weight='bold')

    # double sine
    ax = pl.Subplot(fig, outer[3, 1])
    fig.add_subplot(ax)

    freq1 = 0.1
    freq2 = 5
    amp1s = (0.4, 0.4, 0.7, 0.4, 0.7)
    amp2s = (0.4, 0.4, 0.4, 0.3, 0.5)
    phase_means_data = np.load(os.path.join(save_dir_data_plots, 'sine_stimulus', 'traces', 'rat', 'summary',
                                            'spike_phase',
                                            str(None)+'_'+str(None)+'_'+str(freq1)+'_'+str(freq2),
                                            'phase_means.npy'))
    phase_stds_data = np.load(os.path.join(save_dir_data_plots, 'sine_stimulus', 'traces', 'rat', 'summary',
                                            'spike_phase',
                                            str(None)+'_'+str(None)+'_'+str(freq1)+'_'+str(freq2),
                                            'phase_stds.npy'))

    ax.plot(phase_means_data, phase_stds_data, 'o', color=color_exp, alpha=0.5)

    axins = inset_axes(ax, width='50%', height='50%', loc=1)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    axins.plot(phase_means_data, phase_stds_data, 'o', color=color_exp, alpha=0.5, label='Data')

    for model_idx, model in enumerate(models):
        with open(os.path.join(save_dir_model, model, 'img', 'sine_stimulus/traces',
                               str(amp1s[model_idx]) + '_' + str(amp2s[model_idx]) + '_' + str(freq1) + '_' + str(
                                   freq2),
                               'phase_hist', 'sine_dict.json'), 'r') as f:
            sine_dict_model = json.load(f)
        ax.plot(sine_dict_model['mean_phase'], sine_dict_model['std_phase'], 'o', color=color_model, alpha=0.5)
        axins.plot(sine_dict_model['mean_phase'], sine_dict_model['std_phase'], 'o', color=color_model, alpha=0.5)
        axins.annotate(str(model_idx + 1), xy=(sine_dict_model['mean_phase'][0] + 0.15, sine_dict_model['std_phase'][0] + 1.2), fontsize=8)
    axins.set_xlim(115, 184)
    axins.set_ylim(18, 95)
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)

    ax.set_ylim(0, 360)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_ylabel('Std. phase (deg.)')
    ax.set_xlabel('Mean phase (deg.)')
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.3, 1.0, 'H', transform=ax.transAxes, size=18, weight='bold')

    pl.tight_layout()
    pl.subplots_adjust(bottom=0.05, top=0.98, left=0.11, wspace=0.35)
    pl.savefig(os.path.join(save_dir_img, 'models_compared_in_vitro.pdf'))
    pl.show()