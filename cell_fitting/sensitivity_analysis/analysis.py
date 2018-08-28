import os
import json
import numpy as np
from cell_characteristics.analyze_APs import get_v_rest, get_spike_characteristics, get_AP_onset_idxs
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_fitting.util import init_nan
from cell_characteristics import to_idx


# save dir
folder = '../results/sensitivity_analysis/mean_2std_6models'
#dates = ['2017-10-26_14:13:11']
save_dir_analysis = os.path.join(folder, 'analysis')
dates = filter(lambda x: os.path.isdir(os.path.join(save_dir_analysis, x)), os.listdir(save_dir_analysis))
save_dirs = [os.path.join(folder, date) for date in dates]
return_characteristics = ['AP_amp', 'AP_width', 'fAHP_amp', 'DAP_amp', 'DAP_deflection', 'DAP_width', 'DAP_time']
characteristics_valid_ranges = [(50, 150), (0.1, 2.0), (0, 40), (0, 40), (0, 20), (0, 70), (0, 20)]
v_rest_valid_range = (-90, -60)

spike_characteristics_dict = get_spike_characteristics_dict()

# load params, t, i_inj
with open(os.path.join(save_dirs[0], 'params.json'), 'r') as f:
    params = json.load(f)
with open(os.path.join(save_dirs[0], 't.npy'), 'r') as f:
    t = np.load(f)
with open(os.path.join(save_dirs[0], 'i_inj.npy'), 'r') as f:
    i_inj = np.load(f)

# save
if not os.path.exists(save_dir_analysis):
    os.makedirs(save_dir_analysis)
with open(os.path.join(save_dir_analysis, 'return_characteristics.npy'), 'w') as f:
    np.save(f, return_characteristics)

# analysis
for i_dir, save_dir in enumerate(save_dirs):
    for i_candidate in range(params['n_candidates']):

        # load candidate
        candidate_dir = os.path.join(save_dir, str(i_candidate))
        with open(os.path.join(candidate_dir, 'candidate.npy'), 'r') as f:
            candidate = np.load(f)

        # load voltage
        with open(os.path.join(candidate_dir, 'v.npy'), 'r') as f:
            v = np.load(f)

        # compute features
        dt = t[1] - t[0]
        v_rest = get_v_rest(v, i_inj)
        onsets = get_AP_onset_idxs(v, spike_characteristics_dict['AP_threshold'])
        stim_on = 10
        time_to_AP = 3

        if v_rest_valid_range[0] <= v_rest <= v_rest_valid_range[1] and len(onsets) == 1 \
                and to_idx(stim_on, dt) < onsets[0] < to_idx(stim_on + time_to_AP, dt):
            nonzero = np.nonzero(i_inj)[0]
            if len(nonzero) == 0:
                to_current = -1
            else:
                to_current = nonzero[0] - 1
            std_idx_times = (0, to_current * dt)

            characteristics = np.array(get_spike_characteristics(v, t, return_characteristics, v_rest, check=False
                                                                 **get_spike_characteristics_dict()))

            for i_c, characteristic in enumerate(characteristics):
                if not characteristics_valid_ranges[i_c][0] <= characteristic <= characteristics_valid_ranges[i_c][1]:
                        characteristics = init_nan(len(return_characteristics))
                        break
        else:
            characteristics = init_nan(len(return_characteristics))
        print characteristics

        # save
        candidate_dir_analysis = os.path.join(save_dir_analysis, dates[i_dir], str(i_candidate))
        if not os.path.exists(candidate_dir_analysis):
            os.makedirs(candidate_dir_analysis)
        with open(os.path.join(candidate_dir_analysis, 'characteristics.npy'), 'w') as f:
            np.save(f, characteristics)