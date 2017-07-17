from new_optimization.evaluation.evaluate import get_best_candidate
import json
from new_optimization.fitter import FitterFactory
import matplotlib.pyplot as pl
import numpy as np


def plot(candidate, data_dir):
    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    optimization_settings['fitter_params']['data_dirs'] = [data_dir]
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])

    best_candidate_params = np.array([float(x) for x in candidate.candidate.split()])

    if type(fitter.simulation_params) is list:
        for i, sim_params in enumerate(fitter.simulation_params):
            v_model, t, i_inj = fitter.simulate_cell(best_candidate_params, sim_params)

            best_candidate_params[5] -= best_candidate_params[5] * 0.05
            v_model2, t, i_inj = fitter.simulate_cell(best_candidate_params, sim_params)

            pl.figure()
            pl.plot(fitter.datas[i].t, v_model, 'k', label='no block')
            pl.plot(fitter.datas[i].t, v_model2, 'r', label='Kdr blocked by 5%')
            pl.legend(fontsize=16)
            pl.xlabel('Time (ms)', fontsize=16)
            pl.ylabel('Membrane Potential (mV)', fontsize=16)
            pl.show()
    else:
        v_model, t, i_inj = fitter.simulate_cell(best_candidate_params)
        pl.figure()
        pl.plot(fitter.data.t, v_model, 'k')
        pl.legend(fontsize=16)
        pl.xlabel('Time (ms)', fontsize=16)
        pl.ylabel('Membrane Potential (mV)', fontsize=16)
        pl.show()

if __name__ == '__main__':
    save_dir = '../../results/server/2017-04-14_20:48:28/6/L-BFGS-B'
    method = 'L-BFGS-B'
    n_best = 0
    data_dir = '../../data/2015_08_06d/correct_vrest_-16mV/rampIV/3.5(nA).csv'

    best_candidate = get_best_candidate(save_dir, n_best)
    plot(best_candidate, data_dir)