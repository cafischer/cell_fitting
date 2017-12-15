import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from cell_fitting.sensitivity_analysis import rename_nat_and_nap


if __name__ == '__main__':
    # save dir
    save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'mean_std_6models', 'analysis')
    save_dir_plots = os.path.join(save_dir_analysis, 'plots', 'correlation', 'parameter_characteristic', 'all')

    # load
    with open(os.path.join(save_dir_analysis, 'params.json'), 'r') as f:
        params = json.load(f)
    variable_names = [p[2][0][-2] + ' ' + p[2][0][-1] for p in params['variables']]
    variable_names = rename_nat_and_nap(variable_names)
    return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))
    characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))
    candidate_mat = np.load(os.path.join(save_dir_analysis, 'candidate_mat.npy'))

    # X: (n_samples, n_features)
    n_samples, n_variables = np.shape(candidate_mat)
    X = candidate_mat

    n_samples, n_variables = np.shape(characteristics_mat)
    X = characteristics_mat

    # global centering
    X = X - np.array([np.mean(X, 0)]).repeat(n_samples, axis=0)

    # local centering
    X = X - np.array([np.mean(X, 1)]).T.repeat(n_variables, axis=1)

    # Fit the models
    n_components = np.arange(1, n_variables+1, 4)

    def compute_scores(X):
        pca = PCA(svd_solver='full')
        fa = FactorAnalysis()

        pca_scores, fa_scores = [], []
        for n in n_components:
            pca.n_components = n
            fa.n_components = n
            pca_scores.append(np.mean(cross_val_score(pca, X)))
            fa_scores.append(np.mean(cross_val_score(fa, X)))


        fa.tol = 1e-3
        fa.max_iter = 2000
        fa.n_components = len(variable_names)
        factor = fa.fit(X)
        df = pd.DataFrame(factor.components_, columns=return_characteristics)
        print df

        return pca_scores, fa_scores


    def shrunk_cov_score(X):
        shrinkages = np.logspace(-2, 0, 30)
        cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
        return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


    def lw_score(X):
        return np.mean(cross_val_score(LedoitWolf(), X))


    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    pca = PCA(svd_solver='full', n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    plt.axvline(n_components_pca, color='b',
                label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r',
                label='FactorAnalysis CV: %d' % n_components_fa,
                linestyle='--')
    plt.axvline(n_components_pca_mle, color='k',
                label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

    # compare with other covariance estimators
    plt.axhline(shrunk_cov_score(X), color='violet',
                label='Shrunk Covariance MLE', linestyle='-.')
    plt.axhline(lw_score(X), color='orange',
                label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    plt.show()