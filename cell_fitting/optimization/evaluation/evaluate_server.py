import numpy as np
import pandas as pd
import os
from cell_fitting.optimization.evaluation.evaluate import get_best_candidate, plot_candidate


if __name__ == '__main__':
    #save_dir = '../scripts/test/'
    save_dir = '../../results/server_19_01_09/2019-01-09_17:15:50'  # 2019-01-09_17:15:50  # 2019-01-09_15:47:40
    method = 'L-BFGS-B'
    n_trials = 100
    n_best = 10

    best_candidates = pd.DataFrame()
    for i in range(n_trials):
        print i
        best_candidate = get_best_candidate(os.path.join(save_dir, str(i), method), n_best=0)
        if not best_candidate is None:
            best_candidate.name = i  # will be index later
            best_candidates = best_candidates.append(best_candidate)
    idx_best = np.argsort(best_candidates.fitness.values)[:n_best]
    best_candidate_from_all = best_candidates.iloc[idx_best[0]]

    best_ten = best_candidates.iloc[idx_best]
    print 'Best Trials: ', [c.name for i, c in best_ten.iterrows()] #best_candidate_from_all.name
    best_candidate = plot_candidate(os.path.join(save_dir, str(best_candidate_from_all.name), method),
                                         best_candidate_from_all)