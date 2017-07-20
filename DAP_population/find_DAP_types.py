from __future__ import division
from heka_reader import HekaReader
import os
import matplotlib.pyplot as pl
import numpy as np
from analyze_intracellular.spike_sorting import pca_tranform, k_means_clustering
from cell_characteristics.analyze_APs import get_AP_amp, get_AP_width, get_fAHP_min, get_DAP_max, get_DAP_amp, \
    get_DAP_width, get_fAHP_min_splines, get_DAP_max_splines, get_DAP_deflection


if __name__ == '__main__':

    # save AP_matrix and cells
    AP_matrix = np.load('./results/get_AP_windows/AP_matrix.npy')
    cells_with_AP = np.load('./results/get_AP_windows/cells_with_AP.npy')
    t_window = np.load('./results/get_AP_windows/t_window.npy')
    AP_max = np.load('./results/get_AP_windows/window_before.npy')
    v_rest = np.load('./results/get_AP_windows/v_rest.npy')
    dt = t_window[1] - t_window[0]
    AP_interval = int(round(3/dt))
    DAP_interval = int(round(10/dt))

    # compute characteristics: AP_max, AP_width, DAP_amp, DAP_width
    AP_amp = np.zeros(len(AP_matrix))
    AP_width = np.zeros(len(AP_matrix))
    fAHP_min_idx = np.zeros(len(AP_matrix))
    DAP_max_idx = np.zeros(len(AP_matrix))
    DAP_amp = np.zeros(len(AP_matrix))
    DAP_deflection = np.zeros(len(AP_matrix))
    DAP_width = np.zeros(len(AP_matrix))
    for i, AP_window in enumerate(AP_matrix):
        AP_amp[i] = get_AP_amp(AP_window, AP_max, v_rest)
        AP_width[i] = get_AP_width(AP_window, t_window, 0, AP_max, AP_max+AP_interval, v_rest)

        std = np.std(AP_window[:int(round(2.0 / dt))])  # take first two ms for estimating the std
        w = np.ones(len(AP_window)) / std
        order = int(round(0.3/dt))  # how many points to consider for the minimum
        fAHP_min_idx[i] = get_fAHP_min_splines(AP_window, t_window, AP_max, len(t_window), order=order,
                                               interval=AP_interval, w=w)
        if np.isnan(fAHP_min_idx[i]):
            DAP_max_idx[i] = None
            DAP_amp[i] = None
            DAP_deflection[i] = None
            DAP_width[i] = None
            continue

        order = int(round(2.0 / dt))  # how many points to consider for the minimum
        dist_to_max = int(round(0.5 / dt))
        DAP_max_idx[i] = get_DAP_max_splines(AP_window, t_window, int(fAHP_min_idx[i]), len(t_window), order=order,
                                             interval=DAP_interval, dist_to_max=dist_to_max, w=w)
        if np.isnan(DAP_max_idx[i]):
            DAP_amp[i] = None
            DAP_deflection[i] = None
            DAP_width[i] = None
            continue
        DAP_amp[i] = get_DAP_amp(AP_window, int(DAP_max_idx[i]), v_rest)
        DAP_deflection[i] = get_DAP_deflection(AP_window, int(fAHP_min_idx[i]), int(DAP_max_idx[i]))
        DAP_width[i] = get_DAP_width(AP_window, t_window, int(fAHP_min_idx[i]), int(DAP_max_idx[i]), len(t_window), v_rest)

    for i, AP_window in enumerate(AP_matrix):
        print cells_with_AP[i]
        print 'AP_amp (mV): ', AP_amp[i]
        print 'AP_width (ms): ', AP_width[i]
        if not np.isnan(fAHP_min_idx[i]):
            print 'fAHP_min: (mV): ', AP_window[int(fAHP_min_idx[i])]
            if not np.isnan(DAP_max_idx[i]):
                print 'DAP_amp: (mV): ', DAP_amp[i]
                print 'DAP_width: (ms): ', DAP_width[i]
        pl.figure()
        pl.title(cells_with_AP[i])
        pl.plot(t_window, AP_window)
        pl.plot(t_window[AP_max], AP_window[AP_max], 'or')
        if not np.isnan(fAHP_min_idx[i]):
            pl.plot(t_window[int(fAHP_min_idx[i])], AP_window[int(fAHP_min_idx[i])], 'or')
            if not np.isnan(DAP_max_idx[i]):
                pl.plot(t_window[int(DAP_max_idx[i])], AP_window[int(DAP_max_idx[i])], 'or')
        pl.show()

    AP_matrix_reduced_PCspace = np.vstack((DAP_amp, DAP_deflection, DAP_width)).T
    #AP_matrix_reduced_PCspace = np.vstack((DAP_amp))
    not_nan = np.logical_not(np.any(np.isnan(AP_matrix_reduced_PCspace), 1))
    AP_matrix_reduced_PCspace = AP_matrix_reduced_PCspace[not_nan, :]
    AP_matrix = AP_matrix[not_nan, :]
    n_components = np.shape(AP_matrix_reduced_PCspace)[1]

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
    labels = k_means_clustering(AP_matrix_reduced_PCspace, n_clusters)

    # plot clustering
    fig, ax = pl.subplots(n_dim, n_dim)
    pl.title('Cluster in 2d')
    for i in range(n_dim):
        for j in range(n_dim):
            for l, x in enumerate(AP_matrix_reduced_PCspace):
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
    pl.show()

# TODO: make possible to plot width for checking
# TODO: classification in DAP-nonDAP for DAP database
# TODO: try wavelets for spike classification?
# TODO: clustering were inter distance is maximized better
# TODO: add features: time to DAp and curvature vom peak to rest
# TODO: fAHP_min etc. should return int and named _idx


# big DAP_amp: 2014_01_07b (2015_06_08a, 2014_12_03m, 2013_12_02e)
# small DAP_amp: 2015_04_02n (2015_06_09e, 2014_02_05c)
# small DAP_deflection: 2015_03_30e (2014_11_27f, no DAP_deflection: 2014_01_22g)
# big DAP_width: 2015_08_25b
# late peak: 2014_12_03m (2015_04_02j, 2015_08_25d)
# normal prototype: 2015_05_28c (2015_06_08e, 2015_04_08g)