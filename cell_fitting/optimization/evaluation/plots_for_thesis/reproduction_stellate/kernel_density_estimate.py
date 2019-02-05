import matplotlib.pyplot as pl
import numpy as np
import scipy.stats as st
pl.style.use('paper_subplots')


def compute_kde_and_alpha_hdr(sample, test_value, n_resample=1000000):
    # kernel density estimate
    kernel = st.gaussian_kde(sample)

    # highest density region (HDR)
    pdf_new_samples = kernel.pdf(kernel.resample(n_resample))
    pdf_test_value = kernel.pdf(test_value)[0]
    alpha_hdr = np.mean(pdf_new_samples >= pdf_test_value)
    return kernel, alpha_hdr


def plot_samples_and_kde(samples, test_value, kernel, n_bins=50):
    fig, ax = pl.subplots()
    ax.hist(samples, bins=n_bins, weights=np.ones(len(samples))/float(len(samples)))
    ax.axvline(test_value, 0, 1, color='k')
    x = np.arange(np.min(samples)-np.std(samples), np.max(samples)+np.std(samples), 0.1)
    density_est = kernel(x)
    ax.plot(x, density_est, 'r')
    ax.set_xlabel('Sample values')
    ax.set_ylabel('Frequency')
    pl.tight_layout()


if __name__ == '__main__':
    # surrogate data from single gaussian
    test_value = 165
    mu_true = 170
    cov_true = 15
    samples = st.norm.rvs(loc=mu_true, scale=cov_true, size=20)

    kernel, alpha_hdr = compute_kde_and_alpha_hdr(samples, test_value, n_resample=1000000)
    print 'alpha HDR: ', alpha_hdr

    # plot
    plot_samples_and_kde(samples, test_value, kernel)
    pl.show()