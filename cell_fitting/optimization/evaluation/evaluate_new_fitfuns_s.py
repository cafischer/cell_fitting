from cell_fitting.optimization.evaluation.evaluate import get_best_candidate_new_fitfuns, plot_candidate_on_other_data
import os
import pandas as pd
import json
from nrn_wrapper import load_mechanism_dir


if __name__ == '__main__':
    save_dir_base = '../../results/server/2017-08-23_08:41:41/'
    method = 'L-BFGS-B'
    folder_name = 'candidates_new_fitfuns.csv'
    save_dir = os.path.join(save_dir_base, '0', method)
    n_best = 0

    with open(os.path.join(save_dir_base, str(0), method, 'optimization_settings.json'), 'r') as f:
        optimization_settings = json.load(f)
    load_mechanism_dir(optimization_settings['fitter_params']['mechanism_dir'])

    best_candidates = pd.read_csv(os.path.join(save_dir_base, folder_name))

    best_candidates = best_candidates.sort_values(by='fitness')
    best_candidate = best_candidates.iloc[n_best]

    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv', 'img/rampIV/3.0(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/rampIV/0.5(nA).csv', 'img/rampIV/0.5(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/IV/-0.1(nA).csv', 'img/IV/-0.1(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/IV/0.2(nA).csv', 'img/IV/0.2(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/IV/0.3(nA).csv', 'img/IV/0.3(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/IV/0.4(nA).csv', 'img/IV/0.4(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/IV/0.7(nA).csv', 'img/IV/0.7(nA).png')
    plot_candidate_on_other_data(save_dir, best_candidate,
                                 '../../data/2015_08_26b/vrest-75/IV/1.0(nA).csv', 'img/IV/1.0(nA).png')
