import copy
import os
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.simulate import extract_simulation_params, simulate_currents, iclamp_handling_onset
from cell_fitting.util import merge_dicts, get_channel_dict_for_plotting, get_channel_color_for_plotting
from cell_fitting.read_heka.i_inj_functions import get_i_inj_double_ramp
from cell_fitting.read_heka import get_i_inj_from_function, get_sweep_index_for_amp
pl.style.use('paper')


def plot_currents_on_ax(ax1, channel_list, currents, t, v):
    channel_dict = get_channel_dict_for_plotting()
    channel_color = get_channel_color_for_plotting()

    #t_plot = t - 7
    t_plot = t

    ax2 = ax1.twinx()
    ax2.plot(t_plot, v, 'k', linestyle=':')
    ax2.set_yticks([])

    for i in range(len(channel_list)):
        ax1.plot(t_plot, -1 * currents[i], color=channel_color[channel_list[i]],
                 label=channel_dict[channel_list[i]])
        nan = np.isnan(currents[i])
        if np.any(nan):
            idx_first_nan = np.where(nan)[0][0]
            ax1.plot(np.insert(t_plot[nan], 0, t_plot[idx_first_nan-1]),
                     np.insert(np.zeros(sum(nan)), 0, -1 * currents[i][idx_first_nan-1]),
                     linestyle='--', color=channel_color[channel_list[i]])
        ax1.set_ylabel('Current (mA/cm$^2$)')
        ax1.set_xlabel('Time (ms)')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
    # ax2.set_ylim(-80, -40)
    # ax1.set_ylim(-0.1, 0.1)
    # ax1.set_xlim(0, 55)


if __name__ == '__main__':
    # parameters
    #data_dir = '../../data/cell_csv_data/2015_08_26b/rampIV/3.0(nA).csv'
    #data_dir = '../../data/cell_csv_data/2015_08_26b/rampIV/3.1(nA).csv'
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    #save_dir = '../../results/server_17_12_04/2018-01-05_14:13:33/154/L-BFGS-B'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    channel_dict = get_channel_dict_for_plotting()
    channel_color = get_channel_color_for_plotting()

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # get simulation_params
    # data = pd.read_csv(data_dir)
    # sim_params = {'onset': 200, 'v_init': -75}
    # simulation_params = merge_dicts(extract_simulation_params(data.v.values, data.t.values, data.i.values), sim_params)

    #i_exp = get_i_inj_double_ramp(1.0, 1.0, 7.0, 0.0, 125, -0.05, 2, 15, tstop=500, dt=0.05)
    i_exp = get_i_inj_from_function('IV', [get_sweep_index_for_amp(-0.1, 'IV')], 1000, 0.01)[0]
    #i_exp = get_i_inj_from_function('rampIV', [get_sweep_index_for_amp(3.1, 'rampIV')], 150, 0.01)[0]
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -75, 'tstop': 1000,
                         'dt': 0.01, 'celsius': 35, 'onset': 200}

    cell.soma(.5).hcn_slow.gbar = 0
    cell.soma(.5).nap.gbar = 0
    cell.soma(.5).nat.gbar = 0
    #cell.soma(.5).kdr.gbar = 0
    #cell.soma(.5).g_pas = 0  #0.000430116611589

    # plot currents
    currents, channel_list = simulate_currents(cell, simulation_params, plot=False)
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    fig, ax = pl.subplots()
    plot_currents_on_ax(ax, channel_list, currents, t, v)
    ax.set_ylabel('Current (mA/cm$^2$)')
    pl.tight_layout()
    pl.show()

    fig, ax1 = pl.subplots()
    for i in range(len(channel_list)):
        ax1.plot(t, -1 * currents[i], label=channel_dict[channel_list[i]], color=channel_color[channel_list[i]])
        ax1.set_ylabel('Current (mA/cm$^2$)')
        ax1.set_xlabel('Time (ms)')
    ax2 = ax1.twinx()
    ax2.plot(t, v, 'k', linestyle=':', label='Mem. pot.')
    ax2.set_ylabel('Mem. pot. (mV)')
    ax2.spines['right'].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
    #ax2.set_ylim(-80, -40)
    #ax1.set_ylim(-0.1, 0.1)
    pl.tight_layout()
    pl.show()


    from itertools import combinations
    from cell_fitting.optimization.helpers import get_channel_list

    channel_list = get_channel_list(cell, 'soma')
    len_comb = 2

    pl.figure()
    scale_fac = np.max(np.abs(-1*np.sum(currents))) / np.max(np.abs(np.diff(v)))
    pl.plot(t[1:], np.diff(v) * scale_fac, 'k', label='dV/dt')
    pl.plot(t, -1*np.sum(currents), 'r', label='$-\sum I_{ion}$', linewidth=3)
    for comb in combinations(range(len(channel_list)), len_comb):
        pl.plot(t, -1 * np.sum(np.array([currents[i] for i in comb]), 0),
                label=str([channel_list[i] for i in comb]))
    pl.ylabel('Current', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=10)
    pl.show()

    idx_nap = np.where(np.array(channel_list) == 'nap')[0][0]
    idx_nat = np.where(np.array(channel_list) == 'nat')[0][0]
    idx_kdr = np.where(np.array(channel_list) == 'kdr')[0][0]
    print -1 * (currents[idx_nap] + currents[idx_nat] + currents[idx_kdr])