from __future__ import division
from heka_reader import HekaReader
import os
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize
from analyze_intracellular.spike_sorting import pca_tranform, k_means_clustering
from cell_characteristics.analyze_APs import get_AP_amp, get_AP_width, get_DAP_amp, get_DAP_width, \
    get_fAHP_min_idx_using_splines, get_DAP_max_idx_using_splines, get_DAP_deflection, \
    get_AP_width_idxs, get_DAP_width_idx


def load(save_dir):
    AP_matrix = np.load(os.path.join(save_dir, 'AP_matrix.npy'))
    cells_with_AP = np.load(os.path.join(save_dir, 'cells_with_AP.npy'))
    t_window = np.load(os.path.join(save_dir, 't_window.npy'))
    AP_max_idx = np.load(os.path.join(save_dir, 'window_before.npy'))
    v_rest = np.load(os.path.join(save_dir, 'v_rest.npy'))
    return AP_matrix, cells_with_AP, t_window, AP_max_idx, v_rest


def check_measures():
    global i, AP_window, exp_fit
    for i, AP_window in enumerate(AP_matrix):
        print cells_with_AP[i]
        print 'AP_amp (mV): ', AP_amp[i]
        print 'AP_width (ms): ', AP_width[i]
        if not np.isnan(DAP_max_idx[i]):
            print 'DAP_amp: (mV): ', DAP_amp[i]
            print 'DAP_width: (ms): ', DAP_width[i]
        if not np.isnan(DAP_exp_slope[i]):
            print 'DAP_exp_slope: ', DAP_exp_slope[i]
            print 'DAP_lin_slope: ', DAP_lin_slope[i]
        pl.figure()
        pl.title(cells_with_AP[i])
        pl.plot(t_window, AP_window)
        pl.plot(t_window[AP_max], AP_window[AP_max], 'or', label='AP_max')
        pl.plot(t_window[AP_width_idxs[i, :]], AP_window[AP_width_idxs[i, :]], '-or', label='AP_width')
        if not np.isnan(fAHP_min_idx[i]):
            pl.plot(t_window[int(fAHP_min_idx[i])], AP_window[int(fAHP_min_idx[i])], 'og', label='fAHP')
            if not np.isnan(DAP_max_idx[i]):
                pl.plot(t_window[int(DAP_max_idx[i])], AP_window[int(DAP_max_idx[i])], 'ob', label='DAP_max')
            if not np.isnan(DAP_width_idx[i]):
                pl.plot([t_window[int(fAHP_min_idx[i])], t_window[int(DAP_width_idx[i])]],
                        [AP_window[int(fAHP_min_idx[i])] - (AP_window[int(fAHP_min_idx[i])] - v_rest) / 2,
                         AP_window[int(DAP_width_idx[i])]],
                        '-ob', label='DAP_width')
            if not np.isnan(DAP_width_idx[i]):
                pl.plot([t_window[int(fAHP_min_idx[i])], t_window[int(DAP_width_idx[i])]],
                        [AP_window[int(fAHP_min_idx[i])] - (AP_window[int(fAHP_min_idx[i])] - v_rest) / 2,
                         AP_window[int(DAP_width_idx[i])]],
                        '-ob', label='DAP_width')
            if not np.isnan(slope_start[i]) and not np.isnan(slope_end[i]):
                pl.plot([t_window[int(slope_start[i])], t_window[int(slope_end[i])]],
                        [AP_window[int(slope_start[i])], AP_window[int(slope_end[i])]],
                        '-oy', label='DAP_width')

                def exp_fit(t, a):
                    diff_exp = np.max(np.exp(-t / a)) - np.min(np.exp(-t / a))
                    diff_points = AP_window[int(slope_start[i])] - AP_window[int(slope_end[i])]
                    return (np.exp(-t / a) - np.min(np.exp(-t / a))) / diff_exp * diff_points + AP_window[
                        int(slope_end[i])]

                pl.plot(t_window[int(slope_start[i]): int(slope_end[i])],
                        exp_fit(np.arange(0, len(AP_window[int(slope_start[i]):int(slope_end[i])]), 1) * dt,
                                DAP_exp_slope[i]), 'y')
        pl.legend()
        pl.show()


if __name__ == '__main__':

    # load AP_matrix and cells
    save_dir = './results/get_AP_windows'
    AP_matrix, cells_with_AP, t_window, AP_max, v_rest = load(save_dir)
    dt = t_window[1] - t_window[0]
    AP_interval = int(round(3/dt))
    DAP_interval = int(round(10/dt))

    # compute characteristics: AP_max, AP_width, DAP_amp, DAP_width
    AP_amp = np.zeros(len(AP_matrix))
    AP_width_idxs = np.zeros((len(AP_matrix), 2), dtype=int)
    AP_width = np.zeros(len(AP_matrix))
    fAHP_min_idx = np.zeros(len(AP_matrix))
    DAP_max_idx = np.zeros(len(AP_matrix))
    DAP_amp = np.zeros(len(AP_matrix))
    DAP_deflection = np.zeros(len(AP_matrix))
    DAP_width_idx = np.zeros(len(AP_matrix))
    DAP_width = np.zeros(len(AP_matrix))
    DAP_time = np.zeros(len(AP_matrix))
    slope_start = np.zeros(len(AP_matrix))
    slope_end = np.zeros(len(AP_matrix))
    DAP_exp_slope = np.zeros(len(AP_matrix))
    DAP_lin_slope = np.zeros(len(AP_matrix))
    for i, AP_window in enumerate(AP_matrix):
        AP_amp[i] = get_AP_amp(AP_window, AP_max, v_rest)
        AP_width_idxs[i, :] = get_AP_width_idxs(AP_window, t_window, 0, AP_max, AP_max+AP_interval, v_rest)
        AP_width[i] = get_AP_width(AP_window, t_window, 0, AP_max, AP_max+AP_interval, v_rest)

        std = np.std(AP_window[:int(round(2.0 / dt))])  # take first two ms for estimating the std
        w = np.ones(len(AP_window)) / std
        order = int(round(0.3/dt))  # how many points to consider for the minimum
        fAHP_min_idx[i] = get_fAHP_min_idx_using_splines(AP_window, t_window, AP_max, len(t_window), order=order,
                                               interval=AP_interval, w=w)
        if np.isnan(fAHP_min_idx[i]):
            DAP_max_idx[i] = None
            DAP_amp[i] = None
            DAP_deflection[i] = None
            DAP_width_idx[i] = None
            DAP_width[i] = None
            DAP_time[i] = None
            slope_start[i] = None
            slope_end[i] = None
            DAP_exp_slope[i] = None
            DAP_lin_slope[i] = None
            continue

        order = int(round(2.0 / dt))  # how many points to consider for the minimum
        dist_to_max = int(round(0.5 / dt))
        DAP_max_idx[i] = get_DAP_max_idx_using_splines(AP_window, t_window, int(fAHP_min_idx[i]), len(t_window), order=order,
                                             interval=DAP_interval, dist_to_max=dist_to_max, w=w)
        if np.isnan(DAP_max_idx[i]):
            DAP_amp[i] = None
            DAP_deflection[i] = None
            DAP_width_idx[i] = None
            DAP_width[i] = None
            DAP_time[i] = None
            slope_start[i] = None
            slope_end[i] = None
            DAP_exp_slope[i] = None
            DAP_lin_slope[i] = None
            continue
        DAP_amp[i] = get_DAP_amp(AP_window, int(DAP_max_idx[i]), v_rest)
        DAP_deflection[i] = get_DAP_deflection(AP_window, int(fAHP_min_idx[i]), int(DAP_max_idx[i]))
        DAP_width_idx[i] = get_DAP_width_idx(AP_window, t_window, int(fAHP_min_idx[i]), int(DAP_max_idx[i]),
                                             len(t_window), v_rest)
        DAP_width[i] = get_DAP_width(AP_window, t_window, int(fAHP_min_idx[i]), int(DAP_max_idx[i]),
                                     len(t_window), v_rest)
        DAP_time[i] = t_window[AP_max] - t_window[int(round(DAP_max_idx[i]))]
        if np.isnan(DAP_width_idx[i]):
            slope_start[i] = None
            slope_end[i] = None
            DAP_exp_slope[i] = None
            DAP_lin_slope[i] = None
            continue
        half_fAHP_crossings = np.nonzero(np.diff(np.sign(AP_window[int(DAP_max_idx[i]):len(t_window)]
                                                         - AP_window[int(fAHP_min_idx[i])])) == -2)[0]
        if len(half_fAHP_crossings) == 0:
            slope_start[i] = None
            slope_end[i] = None
            DAP_exp_slope[i] = None
            DAP_lin_slope[i] = None
            continue
        half_fAHP_idx = half_fAHP_crossings[0] + DAP_max_idx[i]
        slope_start[i] = half_fAHP_idx  # int(round(DAP_width_idx[i] - 10/dt))
        slope_end[i] = len(t_window) - 1  # int(round(DAP_width_idx[i] + 20/dt))
        DAP_lin_slope[i] = np.abs((AP_window[int(slope_end[i])] - AP_window[int(slope_start[i])])
                                  / (t_window[int(slope_end[i])] - t_window[int(slope_start[i])]))

        def exp_fit(t, a):
            diff_exp = np.max(np.exp(-t / a)) - np.min(np.exp(-t / a))
            diff_points = AP_window[int(slope_start[i])] - AP_window[int(slope_end[i])]
            return (np.exp(-t / a) - np.min(np.exp(-t / a))) / diff_exp * diff_points + AP_window[int(slope_end[i])]

        DAP_exp_slope[i] = scipy.optimize.curve_fit(exp_fit,
                                                    np.arange(0, len(AP_window[int(slope_start[i]):int(slope_end[i])]), 1) * dt,
                                                    AP_window[int(slope_start[i]):int(slope_end[i])],
                                                    p0=1, bounds=(0, np.inf))[0]

    #check_measures()

    AP_matrix_clustering = np.vstack((AP_amp, AP_width, DAP_amp, DAP_deflection, DAP_width, DAP_time, DAP_lin_slope, DAP_exp_slope)).T
    not_nan = np.logical_not(np.any(np.isnan(AP_matrix_clustering), 1))
    AP_matrix_clustering = AP_matrix_clustering[not_nan, :]
    AP_matrix = AP_matrix[not_nan, :]
    n_components = np.shape(AP_matrix_clustering)[1]

    # PCA on APs
    # n_components = 4
    # AP_matrix_reduced_PCspace, AP_matrix_reduced, pca = pca_tranform(AP_matrix, n_components)
    # print 'Explained variance: ', np.sum(pca.explained_variance_ratio_)

    # pl.figure()
    # pl.title('APs projected back using '+str(n_components)+' components')
    # for AP in AP_matrix_reduced:
    #     pl.plot(t_window, AP)
    # pl.xlabel('Time (ms)')
    # #pl.savefig(os.path.join(folder, 'spike_sorting', 'dim_reduced_APs.png'))
    # pl.show()

    # clustering
    n_dim = n_components
    n_clusters = 5
    labels = k_means_clustering(AP_matrix_clustering, n_clusters)

    # plot clustering
    fig, ax = pl.subplots(n_dim, n_dim)
    pl.title('Cluster in 2d')
    for i in range(n_dim):
        for j in range(n_dim):
            for l, x in enumerate(AP_matrix_clustering):
                ax[i][j].plot(x[i], x[j], 'o', color=str(labels[l]/n_clusters))
                ax[i][j].set_title('component: '+str(i) + ', component: '+str(j))
                #ax.plot(x[i], x[j], 'o', color=str(labels[l] / n_clusters))
                #ax.set_title('component: ' + str(i) + ', component: ' + str(j))
    #pl.savefig(os.path.join(folder, 'spike_sorting', 'cluster_2d.png'))
    pl.show()

    pl.figure()
    pl.title('Mean AP per cluster')
    for c in range(n_clusters):
        pl.plot(t_window, np.mean(AP_matrix[labels == c, :], 0), color=str(c/n_clusters), linewidth=2)
    pl.xlabel('Time (ms)')
    #pl.savefig(os.path.join(folder, 'spike_sorting', 'mean_AP_per_cluster.png'))
    pl.show()

    fig, ax = pl.subplots(1, n_clusters)
    for i, AP_window in enumerate(AP_matrix):
        ax[labels[i]].plot(t_window, AP_window)
        ax[labels[i]].set_ylim(-80, 65)
    pl.show()


# TODO: try wavelets for spike classification?
# TODO: clustering were inter-distance is maximized better


# big DAP_amp: 2014_01_07b (2015_06_08a, 2014_12_03m, 2013_12_02e)
# small DAP_amp: 2015_04_02n (2015_06_09e, 2014_02_05c)
# small DAP_deflection: 2015_03_30e (2014_11_27f, no DAP_deflection: 2014_01_22g)
# big DAP_width: 2015_08_25b
# late peak: 2014_12_03m (2015_04_02j, 2015_08_25d)
# normal prototype: 2015_05_28c (2015_06_08e, 2015_04_08g)