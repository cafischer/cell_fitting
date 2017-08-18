from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
import pywt
from analyze_intracellular.spike_sorting import pca_tranform
from clustering import cluster
from cell_fitting.DAP_population import get_spike_characteristics


def load(save_dir):
    AP_matrix = np.load(os.path.join(save_dir, 'AP_matrix.npy'))
    cells_with_AP = np.load(os.path.join(save_dir, 'cells_with_AP.npy'))
    t_window = np.load(os.path.join(save_dir, 't_window.npy'))
    AP_max_idx = np.load(os.path.join(save_dir, 'window_before.npy'))
    v_rest = np.load(os.path.join(save_dir, 'v_rest.npy'))
    return AP_matrix, cells_with_AP, t_window, AP_max_idx, v_rest


def wavelet_analysis(X, w_type='haar'):
    wavelet = pywt.Wavelet(w_type)
    len_coeff = int(np.floor((len(t_window) + wavelet.dec_len - 1) / 2))
    wavelet_A = np.zeros((len(X), len_coeff))
    wavelet_D = np.zeros((len(X), len_coeff))
    for i, AP_window in enumerate(X):
        wavelet_A[i, :], wavelet_D[i, :] = pywt.dwt(AP_window, wavelet)

    features_per_sample = np.hstack((wavelet_A, wavelet_D))
    return features_per_sample


def PCA_analysis(AP_matrix, n_components, plot=False):
    AP_matrix_reduced_PCspace, AP_matrix_reduced, pca = pca_tranform(AP_matrix, n_components)

    print 'PCA explained variance: ', np.sum(pca.explained_variance_ratio_)
    if plot:
        pl.figure()
        pl.title('APs projected back using ' + str(n_components) + ' components')
        for AP in AP_matrix_reduced:
            pl.plot(t_window, AP)
        pl.xlabel('Time (ms)')
        pl.show()

    return AP_matrix_reduced_PCspace


if __name__ == '__main__':

    # load AP_matrix and cells
    save_dir = './results/get_AP_windows'
    AP_matrix, cells_with_AP, t_window, AP_max, v_rest = load(save_dir)

    # matrix for clustering
    dt = t_window[1] - t_window[0]
    AP_interval = int(round(3 / dt))
    std_idxs = (0, int(round(2.0 / dt)))
    DAP_interval = int(round(10 / dt))
    order_fAHP_min = int(round(0.3 / dt))  # how many points to consider for the minimum
    order_DAP_max = int(round(2.0 / dt))  # how many points to consider for the minimum
    dist_to_DAP_max = int(round(0.5 / dt))


    AP_matrix_clustering_spike = np.vstack((get_spike_characteristics(AP_matrix, t_window, AP_interval, std_idxs,
                                                                      DAP_interval,
                                                                      np.ones(np.shape(AP_matrix)[0])*v_rest,
                                                                      order_fAHP_min,
                                                                      order_DAP_max, dist_to_DAP_max))).T
    AP_matrix_clustering_PCA = PCA_analysis(AP_matrix, n_components=4, plot=False)
    AP_matrix_clustering_wavelet = wavelet_analysis(AP_matrix, w_type='haar')

    AP_matrix_clustering = np.hstack((AP_matrix_clustering_spike, AP_matrix_clustering_PCA,
                                      AP_matrix_clustering_wavelet))

    # remove nans
    not_nan = np.logical_not(np.any(np.isnan(AP_matrix_clustering), 1))
    AP_matrix_clustering = AP_matrix_clustering[not_nan, :]
    AP_matrix = AP_matrix[not_nan, :]

    # clustering
    #method = 'agglomerative'
    #args = {'n_cluster': 10, 'linkage': 'complete'}
    method = 'dbscan'
    args = {'eps': 160, 'min_samples': 3}   # nice cluster: {'eps': 100, 'min_samples': 5}  # standard DAP: {'eps': 100, 'min_samples': 10}
    labels = cluster(AP_matrix_clustering, method, args)
    n_cluster = len(np.unique(labels))
    print ('Number of cluster: ', n_cluster)
    print('Number of unassgined APs: ', np.sum(labels==-1))
    print('Number of assigned APs', np.shape(AP_matrix_clustering)[0] - np.sum(labels==-1))

    # plot clustering
    # n_components = np.shape(AP_matrix_clustering)[1]
    # fig, ax = pl.subplots(n_components, n_components)
    # pl.title('Cluster in 2d')
    # for i in range(n_components):
    #     for j in range(n_components):
    #         for l, x in enumerate(AP_matrix_clustering):
    #             ax[i][j].plot(x[i], x[j], 'o', color=str(labels[l]/n_clusters))
    #             ax[i][j].set_title('component: '+str(i) + ', component: '+str(j))
    #             #ax.plot(x[i], x[j], 'o', color=str(labels[l] / n_clusters))
    #             #ax.set_title('component: ' + str(i) + ', component: ' + str(j))
    # #pl.savefig(os.path.join(folder, 'spike_sorting', 'cluster_2d.png'))
    # pl.show()

    pl.figure()
    pl.title('Mean AP per cluster')
    for c in np.unique(labels):
        pl.plot(t_window, np.mean(AP_matrix[labels == c, :], 0), color=str(c/n_cluster), linewidth=2)
    pl.xlabel('Time (ms)')
    #pl.savefig(os.path.join(folder, 'spike_sorting', 'mean_AP_per_cluster.png'))
    pl.show()

    for c in np.unique(labels):
        pl.figure()
        for i, AP_window in enumerate(AP_matrix[labels == c]):
            pl.plot(t_window, AP_window)
            pl.ylim(-80, 65)
        pl.show()

    fig, ax = pl.subplots(1, n_cluster)
    for i, AP_window in enumerate(AP_matrix):
        ax[labels[i]].plot(t_window, AP_window)
        ax[labels[i]].set_ylim(-80, 65)
    pl.show()