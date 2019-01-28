import matplotlib.pyplot as pl
import numpy as np
import os
import json
import scipy.stats as st
from sklearn.neighbors.kde import KernelDensity
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    color_exp = '#0099cc'
    color_model = 'k'
    freq1 = 0.1
    freq2 = 5

    # load data
    amp1_data = None
    amp2_data = None
    phase_means_data = np.load(os.path.join(save_dir_data_plots, 'sine_stimulus', 'traces', 'rat', 'summary',
                                            'spike_phase', str(amp1_data) + '_' + str(amp2_data) + '_'
                                            + str(freq1) + '_' + str(freq2), 'phase_means.npy'))
    phase_stds_data = np.load(os.path.join(save_dir_data_plots, 'sine_stimulus', 'traces', 'rat', 'summary',
                                           'spike_phase', str(amp1_data) + '_' + str(amp2_data) +'_'
                                           + str(freq1) + '_' + str(freq2), 'phase_stds.npy'))

    # load model
    amp1 = 0.4
    amp2 = 0.4
    with open(os.path.join(save_dir_model, model, 'img', 'sine_stimulus/traces',
                           str(amp1) + '_' + str(amp2) + '_' + str(freq1) + '_' + str(freq2), 'phase_hist',
                           'sine_dict.json'), 'r') as density_est:
        sine_dict_model = json.load(density_est)
    phase_mean_model = sine_dict_model['mean_phase'][0]
    phase_std_model = sine_dict_model['std_phase'][0]

    # kernel density estimate
    data2d = np.vstack([phase_means_data, phase_stds_data])

    # Peform the kernel density estimate
    xmin = np.min(phase_means_data) - 5
    xmax = np.max(phase_means_data) + 10
    ymin = np.min(phase_stds_data) - 5
    ymax = np.max(phase_stds_data) + 10
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 100),
                       np.linspace(ymin, ymax, 100))
    kernel = st.gaussian_kde(data2d)
    positions = np.vstack([X.ravel(), Y.ravel()])
    density_est = np.reshape(kernel(positions).T, X.shape)

    # plot
    fig, ax = pl.subplots()
    ax.plot(phase_means_data, phase_stds_data, 'o', color=color_exp, alpha=0.9)
    ax.plot(phase_mean_model, phase_std_model, 'o', color=color_model, alpha=0.9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # contour
    cfset = ax.contourf(X, Y, density_est, cmap='Blues')
    cset = ax.contour(X, Y, density_est, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)

    ax.set_xlabel('Mean phase (deg.)')
    ax.set_ylabel('Std. phase (deg.)')
    pl.tight_layout()
    # pl.show()

    # surrogate data from single gaussian
    mu_true = np.array([170, 46])
    cov_true = kernel.covariance
    phase_samples = st.multivariate_normal.rvs(mean=mu_true, cov=cov_true, size=1000)
    phase_means_data, phase_stds_data = phase_samples[:, 0], phase_samples[:, 1]

    # # plot
    # fig, ax = pl.subplots()
    # ax.plot(phase_means_data, phase_stds_data, 'o', color=color_exp, alpha=0.9)
    # ax.plot(phase_mean_model, phase_std_model, 'o', color=color_model, alpha=0.9)
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # ax.set_xlabel('Mean phase (deg.)')
    # ax.set_ylabel('Std. phase (deg.)')
    # pl.tight_layout()
    # pl.show()

    # quantile
    x = np.array([phase_mean_model, phase_std_model])
    mus = np.c_[phase_means_data, phase_stds_data]
    sigma2inv = np.linalg.inv(cov_true)
    xminmus = x - mus
    distances = np.sum((xminmus).dot(sigma2inv) * xminmus, axis=1)  # vectorized version of xminus.dot(sigmainv).dot(xminus.T)

    data2d = np.vstack([phase_means_data, phase_stds_data])
    kernel = st.gaussian_kde(data2d)

    quantile = np.prod(1 - st.chi2(2).cdf(distances))
    print 'quantile: ', quantile

    distance_ = (x-mu_true).dot(np.linalg.inv(cov_true)).dot((x-mu_true).T)
    quantile = 1 - st.chi2(2).cdf(distance_)
    print 'quantile (1 gauss): ', quantile


    normal = st.multivariate_normal(mean=mu_true, cov=cov_true)
    pdfs = normal.pdf(mus)
    pdfx = normal.pdf(x)
    print 'quantile (num 1 gauss): ', max(np.mean(pdfs < pdfx), 1.0/len(pdfs))

    # numerical computation of the quantile
    pdfs = kernel.pdf(kernel.resample(100000))
    pdfx = kernel.pdf(x)
    print 'quantile (num): ', max(np.mean(pdfs < pdfx), 1.0/len(pdfs))


