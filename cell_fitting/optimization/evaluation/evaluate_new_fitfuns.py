from cell_fitting.optimization.evaluation.evaluate import get_best_candidate_new_fitfuns, plot_candidate_on_other_data
import os
import pandas as pd
import json
from nrn_wrapper import load_mechanism_dir


if __name__ == '__main__':
    save_dir_base = '../../results/server/2017-08-23_08:41:41/'
    method = 'L-BFGS-B'
    n_trials = 300
    folder_name = 'candidates_new_fitfuns.csv'

    best_candidates = pd.DataFrame()

    with open(os.path.join(save_dir_base, str(0), method, 'optimization_settings.json'), 'r') as f:
        optimization_settings = json.load(f)
    load_mechanism_dir(optimization_settings['fitter_params']['mechanism_dir'])


    for i in range(n_trials):
        save_dir = os.path.join(save_dir_base, str(i), method)

        fitter_params = {
            'name': 'HodgkinHuxleyFitterSeveralDataSeveralFitfuns',
            'variable_keys': [],
            'errfun_name': 'rms',
            'fitfun_names': [['get_v', 'get_DAP'], ['get_v'], ['get_v', 'get_n_spikes']],
            'model_dir': '',
            'mechanism_dir': None,
            'fitnessweights': [[100, 10], [5], [1, 10]],
            'data_dirs': [
                '../../data/2015_08_26b/vrest-75/simulate_rampIV/3.0(nA).csv',
                '../../data/2015_08_26b/vrest-75/plot_IV/-0.1(nA).csv',
                '../../data/2015_08_26b/vrest-75/plot_IV/0.4(nA).csv'
            ],
            'simulation_params': {'celsius': 35, 'onset': 200},
            'args': {}
        }

        sorted_candidates, fitnesses = get_best_candidate_new_fitfuns(save_dir, fitter_params)
        for j in range(5):
            sorted_candidates.fitness.iloc[j] = fitnesses[j]
            best_candidates = best_candidates.append(sorted_candidates.iloc[j])
        best_candidates.to_csv(os.path.join(save_dir_base, folder_name))

    best_candidates = best_candidates.sort_values(by='fitness')
    best_candidate = best_candidates.iloc[0]
    best_candidates.to_csv(os.path.join(save_dir_base, folder_name))

    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/simulate_rampIV/3.0(nA).csv', 'img/simulate_rampIV/3.0(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/simulate_rampIV/0.5(nA).csv', 'img/simulate_rampIV/0.5(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/plot_IV/-0.1(nA).csv', 'img/plot_IV/-0.1(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/plot_IV/0.2(nA).csv', 'img/plot_IV/0.2(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/plot_IV/0.3(nA).csv', 'img/plot_IV/0.3(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/plot_IV/0.4(nA).csv', 'img/plot_IV/0.4(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/plot_IV/0.7(nA).csv', 'img/plot_IV/0.7(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/plot_IV/1.0(nA).csv', 'img/plot_IV/1.0(nA).png')
