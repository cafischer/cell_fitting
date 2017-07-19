from __future__ import division
from heka_reader import HekaReader
import os
import matplotlib.pyplot as pl
import numpy as np
from analyze_intracellular.spike_sorting import pca_tranform, k_means_clustering
from cell_characteristics.analyze_APs import get_AP_amp, get_AP_width, get_fAHP_min, get_DAP_max, get_DAP_amp, get_DAP_width


if __name__ == '__main__':

    # save AP_matrix and cells
    AP_matrix = np.load('./results/get_AP_windows/AP_matrix.npy')
    t_window = np.load('./results/get_AP_windows/t_window.npy')
    AP_max = np.load('./results/get_AP_windows/window_before.npy')
    v_rest = np.load('./results/get_AP_windows/v_rest.npy')
    dt = t_window[1] - t_window[0]
    AP_interval = int(round(2/dt))

    # compute characteristics: AP_max, AP_width, DAP_amp, DAP_width
    AP_amp = np.zeros(len(AP_matrix))
    AP_width = np.zeros(len(AP_matrix))
    fAHP_min = np.zeros(len(AP_matrix))
    DAP_max = np.zeros(len(AP_matrix))
    DAP_amp = np.zeros(len(AP_matrix))
    DAP_width = np.zeros(len(AP_matrix))
    for i, AP_window in enumerate(AP_matrix):
        #i = 45
        #AP_window = AP_matrix[45]
        print i
        AP_amp[i] = get_AP_amp(AP_window, AP_max, v_rest)
        AP_width[i] = get_AP_width(AP_window, t_window, 0, AP_max, AP_max+AP_interval, v_rest)
        fAHP_min[i] = get_fAHP_min(AP_window, AP_max, len(t_window), interval=AP_interval)
        if np.isnan(fAHP_min[i]):
            DAP_max[i] = None
            DAP_amp[i] = None
            DAP_width[i] = None
            continue
        DAP_max[i] = get_DAP_max(AP_window, int(fAHP_min[i]), len(t_window), interval=AP_interval)
        if np.isnan(DAP_max[i]):
            DAP_amp[i] = None
            DAP_width[i] = None
            continue
        DAP_amp[i] = get_DAP_amp(AP_window, int(DAP_max[i]), v_rest)
        DAP_width[i] = get_DAP_width(AP_window, t_window, int(fAHP_min[i]), int(DAP_max[i]), len(t_window), v_rest)


    # for i, AP_window in enumerate(AP_matrix):
    #     print 'AP_amp (mV): ', AP_amp[i]
    #     print 'AP_width (ms): ', AP_width[i]
    #     if not np.isnan(fAHP_min[i]):
    #         print 'fAHP_min: (mV): ', AP_window[int(fAHP_min[i])]
    #         if not np.isnan(DAP_max[i]):
    #             print 'DAP_amp: (mV): ', DAP_amp[i]
    #             print 'DAP_width: (ms): ', DAP_width[i]
        # pl.figure()
        # pl.plot(t_window, AP_window)
        # pl.plot(t_window[AP_max], AP_window[AP_max], 'or')
        # if not np.isnan(fAHP_min[i]):
        #     pl.plot(t_window[int(fAHP_min[i])], AP_window[int(fAHP_min[i])], 'or')
        # pl.show()

    AP_matrix_reduced_PCspace = np.vstack((AP_amp, AP_width, DAP_amp, DAP_width)).T
    not_nan = np.logical_not(np.any(np.isnan(AP_matrix_reduced_PCspace), 1))
    AP_matrix_reduced_PCspace = AP_matrix_reduced_PCspace[not_nan, :]
    AP_matrix = AP_matrix[not_nan, :]
    n_components = 4

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
    n_clusters = 10
    labels = k_means_clustering(AP_matrix_reduced_PCspace, n_clusters)

    # plot clustering
    fig, ax = pl.subplots(n_dim, n_dim)
    pl.title('Cluster in 2d')
    for i in range(n_dim):
        for j in range(n_dim):
            for l, x in enumerate(AP_matrix_reduced_PCspace):
                ax[i][j].plot(x[i], x[j], 'o', color=str(labels[l]/n_clusters))
                ax[i][j].set_title('component: '+str(i) + ', component: '+str(j))
    #pl.savefig(os.path.join(folder, 'spike_sorting', 'cluster_2d.png'))
    pl.show()

    pl.figure()
    pl.title('Mean AP per cluster')
    for c in range(n_clusters):
        pl.plot(t_window, np.mean(AP_matrix[labels == c, :], 0), color=str(c/n_clusters), linewidth=2)
    pl.xlabel('Time (ms)')
    #pl.savefig(os.path.join(folder, 'spike_sorting', 'mean_AP_per_cluster.png'))
    pl.show()


# TODO: plot all individuals belonging to a cluster
# TODO: cubic spline fit for fAHP_min