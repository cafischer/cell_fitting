import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.preprocessing import scale
from cell_fitting.sensitivity_analysis import rename_nat_and_nap
from sklearn.cross_decomposition import PLSCanonical, PLSRegression


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
    return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))
    characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))
    candidate_mat = np.load(os.path.join(save_dir_analysis, 'candidate_mat.npy'))

    # data sets
    n_samples, n_parameters = np.shape(candidate_mat)
    X = candidate_mat
    n_samples, n_features = np.shape(characteristics_mat)
    Y = characteristics_mat

    n_components = 2

    # Dataset based latent variables model
    X_train = X[:n_samples // 2]
    Y_train = Y[:n_samples // 2]
    X_test = X[n_samples // 2:]
    Y_test = Y[n_samples // 2:]

    print("Corr(X)")
    print(np.round(np.corrcoef(X.T), 2))
    print("Corr(Y)")
    print(np.round(np.corrcoef(Y.T), 2))

    # #############################################################################
    # Canonical (symmetric) PLS

    # Transform data
    # ~~~~~~~~~~~~~~
    plsca = PLSCanonical(n_components=2)
    plsca.fit(X_train, Y_train)
    X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
    X_test_r, Y_test_r = plsca.transform(X_test, Y_test)

    # Scatter plot of scores
    # ~~~~~~~~~~~~~~~~~~~~~~
    # 1) On diagonal plot X vs Y scores on each components
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.scatter(X_train_r[:, 0], Y_train_r[:, 0], label="train",
                marker="o", c="b", s=25)
    plt.scatter(X_test_r[:, 0], Y_test_r[:, 0], label="test",
                marker="o", c="r", s=25)
    plt.xlabel("x scores")
    plt.ylabel("y scores")
    plt.title('Comp. 1: X vs Y (test corr = %.2f)' %
              np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1])
    plt.xticks(())
    plt.yticks(())
    plt.legend(loc="best")

    plt.subplot(224)
    plt.scatter(X_train_r[:, 1], Y_train_r[:, 1], label="train",
                marker="o", c="b", s=25)
    plt.scatter(X_test_r[:, 1], Y_test_r[:, 1], label="test",
                marker="o", c="r", s=25)
    plt.xlabel("x scores")
    plt.ylabel("y scores")
    plt.title('Comp. 2: X vs Y (test corr = %.2f)' %
              np.corrcoef(X_test_r[:, 1], Y_test_r[:, 1])[0, 1])
    plt.xticks(())
    plt.yticks(())
    plt.legend(loc="best")

    # 2) Off diagonal plot components 1 vs 2 for X and Y
    plt.subplot(222)
    plt.scatter(X_train_r[:, 0], X_train_r[:, 1], label="train",
                marker="*", c="b", s=50)
    plt.scatter(X_test_r[:, 0], X_test_r[:, 1], label="test",
                marker="*", c="r", s=50)
    plt.xlabel("X comp. 1")
    plt.ylabel("X comp. 2")
    plt.title('X comp. 1 vs X comp. 2 (test corr = %.2f)'
              % np.corrcoef(X_test_r[:, 0], X_test_r[:, 1])[0, 1])
    plt.legend(loc="best")
    plt.xticks(())
    plt.yticks(())

    plt.subplot(223)
    plt.scatter(Y_train_r[:, 0], Y_train_r[:, 1], label="train",
                marker="*", c="b", s=50)
    plt.scatter(Y_test_r[:, 0], Y_test_r[:, 1], label="test",
                marker="*", c="r", s=50)
    plt.xlabel("Y comp. 1")
    plt.ylabel("Y comp. 2")
    plt.title('Y comp. 1 vs Y comp. 2 , (test corr = %.2f)'
              % np.corrcoef(Y_test_r[:, 0], Y_test_r[:, 1])[0, 1])
    plt.legend(loc="best")
    plt.xticks(())
    plt.yticks(())
    plt.show()

    # #############################################################################
    # PLS regression, with multivariate response, a.k.a. PLS2
    pls2 = PLSRegression(n_components=2)
    pls2.fit(X, Y)
    print("Estimated B")
    print(np.round(pls2.coef_, 1))
    pls2.predict(X)