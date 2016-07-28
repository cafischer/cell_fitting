from collections import Iterable
import copy

import numpy as np

__author__ = 'caro'


def gradientdescent(theta_init, learn_rate, fun_gradient, num_iterations, lower_bound, upper_bound, *args, **kwargs):

    theta = copy.copy(theta_init)
    for i in range(num_iterations):
        gradient = fun_gradient(theta, *args, **kwargs)
        theta = theta - learn_rate * gradient
        if isinstance(lower_bound, Iterable):
            theta = np.array([min(max(theta[j], lower_bound[j]), upper_bound[j]) for j in range(len(theta))])
        else:
            theta = np.array([min(max(theta[j], lower_bound), upper_bound) for j in range(len(theta))])
    return theta


def adagrad(theta_init, fun_gradient, num_iterations, lower_bound, upper_bound, gamma=0.9, eps=1e-8, *args, **kwargs):

    theta = copy.copy(theta_init)
    mean_gradient = 0
    mean_dtheta = 0
    rms_mean_dtheta = 1

    for i in range(num_iterations):
        theta_old = theta
        gradient = fun_gradient(theta, *args, **kwargs)

        mean_gradient = gamma * mean_gradient + (1-gamma) * gradient**2
        rms_mean_gradient = np.sqrt(mean_gradient + eps)

        theta = theta - (rms_mean_dtheta / rms_mean_gradient) * gradient
        if isinstance(lower_bound, Iterable):
            theta = np.array([min(max(theta[j], lower_bound[j]), upper_bound[j]) for j in range(len(theta))])
        else:
            theta = np.array([min(max(theta[j], lower_bound), upper_bound) for j in range(len(theta))])

        mean_dtheta = gamma * mean_dtheta + (1-gamma) * (theta-theta_old)**2
        rms_mean_dtheta = np.sqrt(mean_dtheta + eps)
    return theta


def adagrad_single(theta_init, fun_gradient, num_iterations, lower_bound, upper_bound, gamma=0.9, eps=1e-8, *args, **kwargs):

    theta = copy.copy(theta_init)
    mean_gradient = 0
    mean_dtheta = 0
    rms_mean_dtheta = 1

    for i in range(num_iterations):
        for j in range(len(theta)):
            mask = np.zeros(len(theta), dtype=bool)
            mask[j] = 1
            theta_old = theta
            gradient = fun_gradient(theta, *args, **kwargs)

            print 'gradient: ' + str(gradient)
            print 'mask: ' + str(mask)  # TODO
            print 'theta: ' + str(theta)
            print  # TODO

            mean_gradient = gamma * mean_gradient + (1-gamma) * gradient**2
            rms_mean_gradient = np.sqrt(mean_gradient + eps)

            theta[mask] = theta[mask] - (rms_mean_dtheta / rms_mean_gradient)[mask] * gradient[mask]
            if isinstance(lower_bound, Iterable):
                theta[mask] = min(max(theta[mask], lower_bound[mask]), upper_bound[mask])
            else:
                theta[mask] = min(max(theta[mask], lower_bound), upper_bound)

            mean_dtheta = gamma * mean_dtheta + (1-gamma) * (theta-theta_old)**2
            rms_mean_dtheta = np.sqrt(mean_dtheta + eps)
    return theta


def gradientdescent_single(theta_init, learn_rate, fun_gradient, num_iterations, lower_bound, upper_bound,
                           *args, **kwargs):

    theta = copy.copy(theta_init)
    for i in range(num_iterations):
        for j in range(len(theta)):
            mask = np.zeros(len(theta), dtype=bool)
            mask[j] = 1
            gradient = fun_gradient(theta, *args, **kwargs)
            theta[mask] = theta[mask] - learn_rate * gradient[mask]
            if isinstance(lower_bound, Iterable):
                theta[mask] = min(max(theta[mask], lower_bound[mask]), upper_bound[mask])
            else:
                theta[mask] = min(max(theta[mask], lower_bound), upper_bound)

            print 'gradient: ' + str(gradient)
            print 'mask: ' + str(mask)  # TODO
            print 'theta: ' + str(theta)
            print  # TODO
    return theta


def numerical_gradient(x, f, h=1e-5, method='central', *args, **kwargs):
    """
    Numerical gradient of f at x based on forward difference.

    :param f: Function with one argument (array for higher dimensionality).
    :type f: function
    :param x: The point to evaluate the gradient at.
    :type x: array_like
    :param h: Distance to the next point. (The smaller the better the approximation of the gradient.)
    :type h: float
    :return: Numerical gradient (forward difference) of f at x.
    :rtype: array_like
    """

    gradient = np.zeros(len(x))

    # iterate over all dimensions of x
    for i in range(len(x)):

        # compute the partial derivative
        mask = np.zeros(len(x), dtype=bool)
        mask[i] = 1

        if method == 'forward':
            gradient[mask] = (f(x + mask*h, *args, **kwargs) - f(x, *args, **kwargs)) / h
        elif method == 'central':
            gradient[mask] = (f(x + 0.5*mask*h, *args, **kwargs) - f(x - 0.5*mask*h, *args, **kwargs)) / h

    #print 'candidate: '+str(x)
    #print 'gradient: '+str(gradient)
    return gradient




#######################################################################
if __name__ == '__main__':

    # test numerical_gradient
    def f(x):
        return np.array([x[0]**2 + x[1]])

    x = np.array([2, 4])
    h = 1

    gradient = numerical_gradient(x, f, h)

    if gradient[0] == (f(np.array([3, 4]))-f(x))/h and gradient[1] == (f(np.array([2, 5]))-f(x))/h:
        print 'Test successful!'