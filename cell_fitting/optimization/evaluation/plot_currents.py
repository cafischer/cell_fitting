import copy
import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from nrn_wrapper import Cell

from cell_fitting.optimization.simulate import extract_simulation_params, simulate_currents, iclamp_handling_onset
from cell_fitting.util import merge_dicts

pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    data_dir = '../../data/2015_08_26b/vrest-75/simulate_rampIV/3.0(nA).csv'
    #data_dir = '../../data/2015_08_26b/vrest-75/IV/0.4(nA).csv'
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # get simulation_params
    data = pd.read_csv(data_dir)
    sim_params = {'onset': 200, 'v_init': -75}
    simulation_params = merge_dicts(extract_simulation_params(data), sim_params)

    # plot currents
    currents, channel_list = simulate_currents(cell, simulation_params, plot=False)
    new_channel_list = copy.copy(channel_list)
    new_channel_list[channel_list.index('nap')] = 'nat'
    new_channel_list[channel_list.index('nat')] = 'nap'
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    fig, ax1 = pl.subplots()
    for i in range(len(channel_list)):
        ax1.plot(t, -1 * currents[i], label=new_channel_list[i])
        ax1.set_ylabel('Current (mA/cm$^2$)', fontsize=16)
        ax1.set_xlabel('Time (ms)', fontsize=16)
    ax2 = ax1.twinx()
    ax2.plot(t, v, 'k', label='Mem. Pot.')
    ax2.set_ylabel('Membrane Potential (mV)', fontsize=16)
    ax2.spines['right'].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=16)
    pl.tight_layout()
    pl.show()


    from itertools import combinations
    from cell_fitting.optimization.helpers import get_channel_list

    channel_list = get_channel_list(cell, 'soma')
    len_comb = 2

    pl.figure()
    scale_fac = np.max(np.abs(-1*np.sum(currents))) / np.max(np.abs(np.diff(data.v)))
    pl.plot(data.t[1:], np.diff(data.v) * scale_fac, 'k', label='dV/dt')
    pl.plot(data.t, -1*np.sum(currents), 'r', label='$-\sum I_{ion}$', linewidth=3)
    for comb in combinations(range(len(channel_list)), len_comb):
        pl.plot(data.t, -1 * np.sum(np.array([currents[i] for i in comb]), 0),
                label=str([channel_list[i] for i in comb]))
    pl.ylabel('Current', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=10)
    pl.show()
