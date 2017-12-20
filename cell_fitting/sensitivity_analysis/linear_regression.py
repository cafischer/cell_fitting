import numpy as np
import matplotlib.pyplot as plt
import os
import json
from cell_fitting.sensitivity_analysis import rename_nat_and_nap
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model


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

    # linear regression
    lr = linear_model.LinearRegression()
    Y = characteristics_mat
    X = candidate_mat

    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:
    predicted = cross_val_predict(lr, X, Y, cv=10)

    lr.fit(X, Y)
    print 'Score: '+ str(lr.score(X, Y))  # 1 is best, can be arbitrarily negative

    fig, ax = plt.subplots()
    ax.scatter(Y, predicted, edgecolors=(0, 0, 0))
    ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()