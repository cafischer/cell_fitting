import numpy as np
import pandas as pd
import os
from new_optimization.evaluation.evaluate import get_best_candidate, plot_candidate


if __name__ == '__main__':
    #save_dir = '../scripts/test/'
    save_dir = '../../results/server/2017-08-16_09:41:34'
    method = 'L-BFGS-B'
    n_trials = 170
    n_best = 0

    best_candidates = pd.DataFrame()
    for i in range(n_trials):
        print i
        best_candidate = get_best_candidate(os.path.join(save_dir, str(i), method), n_best=0)
        if not best_candidate is None:
            best_candidate.name = i  # will be index later
            best_candidates = best_candidates.append(best_candidate)
    idx_best = np.argsort(best_candidates.fitness.values)[n_best]
    best_candidate_from_all = best_candidates.iloc[idx_best]

    best_ten = best_candidates.iloc[np.argsort(best_candidates.fitness.values)[:10]]
    print 'Best Trials: ', [c.name for i, c in best_ten.iterrows()] #best_candidate_from_all.name
    best_candidate = plot_candidate(os.path.join(save_dir, str(best_candidate_from_all.name), method),
                                         best_candidate_from_all)