import os
import json
import numpy as np
from cell_characteristics.analyze_APs import get_v_rest, get_spike_characteristics
from cell_fitting.util import init_nan
import matplotlib.pyplot as pl
pl.style.use('paper')


# save dir
dates = ['2017-10-06_09:04:12', '2017-10-06_09:49:29']
save_dirs = [os.path.join('../results/sensitivity_analysis/', date) for date in dates]
save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'analysis_1')
return_characteristics = ['AP_amp', 'AP_width', 'DAP_amp', 'DAP_deflection', 'DAP_width', 'DAP_time',
                          'DAP_lin_slope', 'DAP_exp_slope']
characteristics_valid_ranges = [(90, 140), (0.3, 2.5), (10, 80), (0.1, 20), (5, 50), (2, 30), (None, None), (None, None)]

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

            #for i, var in enumerate(params['variables']):
            #    print str(var[2]) + ': ' + str(candidate[i])

        # load voltage
        with open(os.path.join(candidate_dir, 'v.npy'), 'r') as f:
            v = np.load(f)

        # compute features
        dt = t[1] - t[0]
        v_rest = get_v_rest(v, i_inj)

        nonzero = np.nonzero(i_inj)[0]
        if len(nonzero) == 0:
            to_current = -1
        else:
            to_current = nonzero[0] - 1
        characteristics = get_spike_characteristics(v, t, return_characteristics, v_rest, AP_threshold=-30,
                                                    AP_interval=4, std_idx_times=(0, to_current * dt), k_splines=5,
                                                    s_splines=0, order_fAHP_min=None, DAP_interval=40,
                                                    order_DAP_max=None, min_dist_to_DAP_max=0, check=False)

        for i_c, characteristic in enumerate(characteristics):
            if not characteristics_valid_ranges[i_c][0] <= characteristic <= characteristics_valid_ranges[i_c][1]:
                characteristics[i_c:] = init_nan((len(characteristics)-i_c))
                break
        print characteristics

        # save
        candidate_dir_analysis = os.path.join(save_dir_analysis, dates[i_dir], str(i_candidate))
        if not os.path.exists(candidate_dir_analysis):
            os.makedirs(candidate_dir_analysis)
        with open(os.path.join(candidate_dir_analysis, 'characteristics.npy'), 'w') as f:
            np.save(f, characteristics)

        # pl.figure()
        # pl.plot(t, v)
        # pl.show()
        #
        # pl.figure()
        # pl.plot(t, i_inj)
        # pl.show()