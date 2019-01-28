import numpy as np
import os
import json
import matplotlib.pyplot as pl
from cell_fitting.optimization.evaluation.plot_sine_stimulus import get_sine_stimulus
from grid_cell_stimuli.spike_phase import get_spike_phases_by_min
from cell_characteristics import to_idx
from sklearn.metrics import mean_squared_error


def get_poisson_rmse():
    spike_train = firing_rate * dt_data > np.random.rand(len(t))

    # spike_train = np.zeros(len(t))
    # refractory = 0
    # for i, t_step in enumerate(t):
    #     if firing_rate[i] * dt_data > np.random.rand() and refractory == 0:
    #         spike_train[i] = 1
    #         refractory = 2  # ms refractory
    #     if refractory > 0:
    #         refractory -= dt_data
    #     if refractory < 0:
    #         refractory = 0

    # compute phases
    phases_poisson = get_spike_phases_by_min(np.where(spike_train)[0], t, theta, order, dist_to_AP)
    phases_poisson = phases_poisson[~np.isnan(phases_poisson)]

    # compute phase histogram
    hist_poisson, _ = np.histogram(phases_poisson, bins)

    # plots
    # pl.figure()
    # pl.plot(t[spike_train.astype(bool)], i_inj_data[spike_train.astype(bool)], 'or')
    # pl.plot(t, i_inj_data, 'k')
    #
    # pl.figure()
    # pl.bar(bins[:-1], hist_data, width=bin_width, color='k', alpha=0.5)
    # pl.bar(bins[:-1], hist_poisson, width=bin_width, color='r', alpha=0.5)
    # pl.show()

    # compute dissimilarity to exp. cell phase histogram (rmse)
    rmse = np.sqrt(mean_squared_error(hist_poisson, hist_data))

    return rmse, len(phases_poisson)


if __name__ == '__main__':
    # parameters
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    bin_width = 10
    bins=np.arange(0, 360 + bin_width, bin_width)
    n_poisson = 10000
    np.random.seed(1)

    # get input stimulus
    amp1_data = 0.4
    amp2_data = 0.2
    freq1 = 0.1
    freq2 = 5
    sine1_dur = 1./freq1 * 1000 / 2.
    dt_data = 0.05
    tstop = 5999.95
    onset_dur = offset_dur = 500
    i_inj_data = get_sine_stimulus(amp1_data, amp2_data, 1./freq1*1000/2., freq2, 500, 500-dt_data, dt_data)

    # get theta oscillation
    x = np.arange(0, sine1_dur + dt_data, dt_data)
    theta = amp2_data * np.sin(2 * np.pi * x * freq2 / 1000.)
    onset = np.zeros(to_idx(onset_dur, dt_data))
    offset = np.zeros(to_idx(offset_dur, dt_data))[:-1]
    theta = np.concatenate((onset, theta, offset))
    # parameters for phase computation
    order = to_idx(20, dt_data)
    dist_to_AP = to_idx(1. / freq2 * 1000, dt_data)

    # phase hist data and model
    amp1 = 0.4
    amp2 = 0.4
    model = '2'
    with open(os.path.join(save_dir_model, model, 'img', 'sine_stimulus/traces',
                           str(amp1) + '_' + str(amp2) + '_' + str(freq1) + '_' + str(freq2), 'phase_hist',
                           'sine_dict.json'), 'r') as f:
        sine_dict_model = json.load(f)
    phases_model = sine_dict_model['phases']
    hist_model, _ = np.histogram(phases_model, bins)

    with open(os.path.join(save_dir_data_plots, 'sine_stimulus/traces/rat', '2015_08_20d',  # using different cell here!
                           str(amp1_data) + '_' + str(amp2_data) + '_' + str(freq1) + '_' + str(freq2),
                           'spike_phase', 'sine_dict.json'), 'r') as f:
        sine_dict_data = json.load(f)
    phases_data = sine_dict_data['phases']
    hist_data, _ = np.histogram(phases_data, bins)

    # compute Poisson spike-trains, use double-sine as firing rate (normalize so that average firing rate fits to exp. cell)
    firing_rate = i_inj_data / np.mean(i_inj_data) * (len(phases_data) / tstop)
    t = np.arange(0, tstop + dt_data, dt_data)

    rmses = np.zeros(n_poisson)
    n_phasess = np.zeros(n_poisson)
    for i in range(n_poisson):
        rmses[i], n_phasess[i] = get_poisson_rmse()

    rmse_model = np.sqrt(mean_squared_error(hist_model, hist_data))

    # check for significance (H0: RMSE_model >= RMSE_PoissonDistribution)
    alpha = 0.01
    percentile = alpha * 100.
    assert percentile / 100. * n_poisson >= 10
    test_val = np.percentile(rmses, percentile)

    print 'RMSE (Percentile: '+str(percentile)+')', test_val
    print 'RMSE (Percentile: ' + str(0.1) + ')', np.percentile(rmses, 0.1)
    print 'RMSE (Percentile: ' + str(0.01) + ')', np.percentile(rmses, 0.01)
    print 'p-val: ', 1 - max(np.mean(rmse_model < rmses), 1./len(rmses))
    print 'Significant: ', rmse_model < test_val
    print '# Phases (Poisson)', np.mean(n_phasess)
    print '# Phases (Data)', len(phases_data)

    # plot distribution of similarity values, plot similarity value of the model
    pl.figure()
    pl.hist(rmses, bins=50, color='r', alpha=0.5)
    pl.axvline(rmse_model, 0, 1, color='b')
    pl.ylabel('Frequency')
    pl.xlabel('RMSE')
    pl.show()

    # np.random.seed(1)
    # RMSE(Percentile: 1.0) 2.04124145232
    # RMSE(Percentile: 0.1) 1.81045572742
    # RMSE(Percentile: 0.01) 1.74801395058
    # p-val: 0.0008 (1 - 0.9992)
    # Significant:  True
    # # Phases (Poisson) 84.7244
    # # Phases (Data) 83