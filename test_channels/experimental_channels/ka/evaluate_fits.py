import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import json
from new_optimization.fitter import FitterFactory


def get_best_candidate(save_dir, n_best):
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    candidates_best = pd.DataFrame(columns=candidates.columns)
    for id in np.unique(candidates.id):
        candidates_id = candidates[candidates.id == id]
        try:
            candidates_best = candidates_best.append(candidates_id.iloc[np.nanargmin(candidates_id.fitness.values)])
        except ValueError:
            pass
    idx_best = np.argsort(candidates_best.fitness.values)[n_best]
    return candidates_best.iloc[idx_best]


def plot_candidate(save_dir, n_best):
    best_candidate = get_best_candidate(save_dir, n_best)

    best_candidate_x = np.array([float(x) for x in best_candidate.candidate.split()])
    print 'id: ' + str(best_candidate.id)
    print 'generation: ' + str(best_candidate.generation)
    print 'fitness: ' + str(best_candidate.fitness)
    print 'candidate: ' + str(best_candidate_x)

    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    i_channel = fitter.simulate(best_candidate_x, None)

    pl.figure()
    for i, v in enumerate(fitter.v_steps):
        pl.plot(fitter.i_traces.index, fitter.i_traces[str(v)], 'k', label='Data')
        pl.plot(fitter.i_traces.index, i_channel[i], 'r', label='Model')
    #pl.legend(fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Current (normalized)', fontsize=16)
    pl.savefig(save_dir + '/best_candidate.png')
    pl.show()


if __name__ == '__main__':

    save_dir = '/media/caro/Daten/Phd/DAP-Project/cell_fitting/results/ion_channels/ka/L-BFGS-B'
    plot_candidate(save_dir, 0)
