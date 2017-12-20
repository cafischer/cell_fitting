import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from cell_fitting.sensitivity_analysis import rename_nat_and_nap


if __name__ == '__main__':
    # save dir
    save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'mean_std_6models_nr1', 'analysis')
    save_dir_img = os.path.join(save_dir_analysis, 'plots', 'factor_analysis_and_PCA')

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # load
    with open(os.path.join(save_dir_analysis, 'params.json'), 'r') as f:
        params = json.load(f)
    variable_names = [p[2][0][-2] + ' ' + p[2][0][-1] for p in params['variables']]
    variable_names = rename_nat_and_nap(variable_names)
    #return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))
    #characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))
    candidate_mat = np.load(os.path.join(save_dir_analysis, 'candidate_mat.npy'))

    # X: (n_samples, n_features)
    n_samples, n_variables = np.shape(candidate_mat)
    X = candidate_mat

    #n_samples, n_variables = np.shape(characteristics_mat)
    #X = characteristics_mat

    # normalization
    X = scale(X, axis=0)  # axis = 0 normalize feature

    # Fit the models
    n_components = np.arange(1, n_variables+1, 1)

    def compute_scores(X):
        pca = PCA(svd_solver='full')
        fa = FactorAnalysis()

        pca_scores, fa_scores = [], []
        for n in n_components:
            pca.n_components = n
            fa.n_components = n
            pca_scores.append(np.mean(cross_val_score(pca, X)))
            fa_scores.append(np.mean(cross_val_score(fa, X)))
        return pca_scores, fa_scores


    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    print("Best n_components by PCA CV = %d" % n_components_pca)
    print("Best n_components by FactorAnalysis CV = %d" % n_components_fa)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA')
    plt.plot(n_components, fa_scores, 'r', label='FA')
    plt.xlabel('Number of Components')
    plt.ylabel('CV Scores')
    plt.legend()
    plt.savefig(os.path.join(save_dir_img, 'scores.png'))
    #plt.show()

    # do PCA, Factor Analysis with all components
    pd.options.display.float_format = '{:,.2f}'.format
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()
    print 'Factor Analysis'
    fa.tol = 1e-3
    fa.max_iter = 2000
    fa.n_components = n_variables
    fit = fa.fit(X)
    df = pd.DataFrame(fit.components_, columns=variable_names)
    print df
    df.to_csv(os.path.join(save_dir_img, 'fa_factor_loadings.csv'))

    print 'PCA'
    pca.tol = 1e-3
    pca.max_iter = 2000
    pca.n_components = n_variables
    fit = pca.fit(X)
    df = pd.DataFrame(fit.components_, columns=variable_names)
    print df
    df.to_csv(os.path.join(save_dir_img, 'pca_eigenvectors.csv'))

    plt.figure()
    plt.plot(n_components, pca.explained_variance_ratio_, 'b', label='')
    plt.plot(n_components, np.cumsum(pca.explained_variance_ratio_), 'r', label='cumulative')
    plt.xlabel('Number of Components')
    plt.ylabel('PCA Explained Variance')
    plt.savefig(os.path.join(save_dir_img, 'explained_var.png'))
    plt.legend()

    plt.show()