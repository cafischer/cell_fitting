import os
import pandas as pd
from nrn_wrapper import Cell
from optimization.simulate import extract_simulation_params, simulate_currents
from util import merge_dicts


if __name__ == '__main__':
    # parameters
    data_dir = '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv'
    save_dir = '../../results/server/2017-07-06_13:50:52/434/L-BFGS-B/'
    model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #model_dir = '../../results/server/2017-07-06_13:50:52/434/L-BFGS-B/model/best_cell.json'
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # get simulation_params
    data = pd.read_csv(data_dir)
    sim_params = {'onset': 200, 'v_init': -80}
    simulation_params = merge_dicts(extract_simulation_params(data), sim_params)

    # plot currents
    currents = simulate_currents(cell, simulation_params, plot=True)


    import matplotlib.pyplot as pl
    import numpy as np
    from itertools import combinations
    from optimization.helpers import get_channel_list

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
