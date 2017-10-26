import os
import json
import numpy as np


# save dir
dates = ['2017-10-26_14:13:11']
save_dirs = [os.path.join('../results/sensitivity_analysis/', date) for date in dates]
save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'analysis_test')
save_dir_plots = os.path.join(save_dir_analysis, 'plots')

# load
with open(os.path.join(save_dirs[0], 'params.json'), 'r') as f:
    params = json.load(f)
i_inj = np.load(os.path.join(save_dirs[0], 'i_inj.npy'))
return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))

# build candidate_mat and characteristics_mat
n_variables = len(params['variables'])
n_total_candidates = params['n_candidates']*len(save_dirs)
candidate_mat = np.zeros((n_total_candidates, n_variables))
characteristics_mat = np.zeros((n_total_candidates, len(return_characteristics)))

for i_dir, save_dir in enumerate(save_dirs):
    for i_candidate in range(params['n_candidates']):
        candidate_dir = os.path.join(save_dir, str(i_candidate))
        candidate_mat[i_dir*params['n_candidates']+i_candidate, :] = np.load(os.path.join(candidate_dir,
                                                                                          'candidate.npy'))

        candidate_dir_analysis = os.path.join(save_dir_analysis, dates[i_dir], str(i_candidate))
        characteristics_mat[i_dir*params['n_candidates']+i_candidate, :] = np.load(os.path.join(candidate_dir_analysis,
                                                                                                'characteristics.npy'))

# remove candidates with nan characteristics
candidates_not_nan = ~np.any(np.isnan(characteristics_mat), 1)
characteristics_mat = characteristics_mat[candidates_not_nan, :]
candidate_mat = candidate_mat[candidates_not_nan, :]

# save
np.save(os.path.join(save_dir_analysis, 'characteristics_mat.npy'), characteristics_mat)
np.save(os.path.join(save_dir_analysis, 'candidate_mat.npy'), candidate_mat)
np.savetxt(os.path.join(save_dir_analysis, 'n_candidates_not_nan.txt'), np.array([np.shape(candidate_mat)[0]]))