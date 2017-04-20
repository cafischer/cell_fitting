import json
import pandas as pd
from new_optimization.fitter import FitterFactory
from optimization.simulate import extract_simulation_params, simulate_gates
from evaluate import get_best_candidate, get_candidate_params


if __name__ == '__main__':
    save_dir = '../../results/server/2017-04-11_20:44:13/34/'
    #save_dir = '../../results/new_optimization/2015_08_06d/27_03_17_readjust/'
    method = 'L-BFGS-B'
    n_best = 0
    data_dir = '../../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(3)/0(nA).csv'
    #data_dir = '../../data/2015_08_06d/correct_vrest_-16mV/rampIV/3.5(nA).csv'
    #data_dir = '../../data/2015_08_06d/raw/IV/-0.15(nA).csv'
    #data_dir = '../../data/2015_08_26b/corrected_vrest2/rampIV/3.0(nA).csv'
    #data_dir = '../../data/2015_08_06d/correct_vrest_-16mV/IV/-0.15(nA).csv'

    best_candidate = get_candidate_params(get_best_candidate(save_dir+method+'/', n_best))

    # recover cell and update with best candidate
    with open(save_dir+method+'/' + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    fitter.update_cell(best_candidate)

    # get simulation_params
    data = pd.read_csv(data_dir)
    simulation_params = extract_simulation_params(data)

    # plot currents
    simulate_gates(fitter.cell, simulation_params, plot=True)