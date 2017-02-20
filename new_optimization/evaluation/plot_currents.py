import json
from new_optimization.fitter import *
from evaluate import *


if __name__ == '__main__':
    #save_dir = '../../results/new_optimization/2015_08_06d/16_02_17_PP(4)/'
    save_dir = '../../results/new_optimization/2015_08_06d/16_02_17_PP(4)/'
    method = 'L-BFGS-B'
    n_best = 1
    #data_dir = '../../data/2015_08_06d/raw/PP(4)/0(nA).csv'
    #data_dir = '../../data/2015_08_06d/raw/rampIV/3.5(nA).csv'
    #data_dir = '../../data/2015_08_06d/raw/IV/-0.15(nA).csv'
    #data_dir = '../../data/2015_08_26b/corrected_vrest2/rampIV/3.0(nA).csv'
    data_dir = '../../data/2015_08_26b/corrected_vrest2/IV/-0.15(nA).csv'

    best_candidate = get_best_candidate(save_dir+method+'/', n_best)

    # recover cell and update with best candidate
    with open(save_dir+method+'/' + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    fitter.update_cell(best_candidate)

    # get simulation_params
    data = pd.read_csv(data_dir)
    simulation_params = extract_simulation_params(data)

    # plot currents
    simulate_currents(fitter.cell, simulation_params, plot=True)