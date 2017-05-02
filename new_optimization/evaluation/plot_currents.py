import json
import pandas as pd
from new_optimization.fitter import FitterFactory
from optimization.simulate import extract_simulation_params, simulate_currents
from evaluate import get_best_candidate, get_candidate_params


if __name__ == '__main__':
    save_dir = '../../results/server/2017-05-01_11:03:22/308/'
    #save_dir = '../../results/new_optimization/2015_08_06d/27_03_17_readjust/'
    method = 'L-BFGS-B'
    n_best = 0
    #data_dir = '../../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(3)/0(nA).csv'
    data_dir = '../../data/2015_08_06d/correct_vrest_-16mV/rampIV/3.5(nA).csv'
    #data_dir = '../../data/2015_08_06d/raw/IV/-0.15(nA).csv'
    #data_dir = '../../data/2015_08_26b/corrected_vrest2/rampIV/3.0(nA).csv'
    #data_dir = '../../data/2015_08_06d/correct_vrest_-16mV/IV/-0.15(nA).csv'

    best_candidate_params = get_candidate_params(get_best_candidate(save_dir + method + '/', n_best))
    #best_candidate_params[5] -= best_candidate_params[5] * 0.05

    # recover cell and update with best candidate
    with open(save_dir+method+'/' + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    fitter.update_cell(best_candidate_params)

    # get simulation_params
    data = pd.read_csv(data_dir)
    simulation_params = extract_simulation_params(data)

    # plot currents
    currents = simulate_currents(fitter.cell, simulation_params, plot=True)

    import matplotlib.pyplot as pl
    import numpy as np
    from itertools import combinations
    from optimization.helpers import get_channel_list

    channel_list = get_channel_list(fitter.cell, 'soma')
    len_comb = 2

    pl.figure()
    scale_fac = np.max(np.abs(-1*np.sum(currents))) / np.max(np.abs(np.diff(fitter.data.v)))
    pl.plot(fitter.data.t[1:], np.diff(fitter.data.v) * scale_fac, 'k', label='dV/dt')
    pl.plot(fitter.data.t, -1*np.sum(currents), 'r', label='$-\sum I_{ion}$')
    for comb in combinations(range(len(channel_list)), len_comb):
        pl.plot(fitter.data.t, -1 * np.sum(np.array([currents[i] for i in comb]), 0),
                label=str([channel_list[i] for i in comb]))
    pl.ylabel('Current', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=10)
    pl.show()