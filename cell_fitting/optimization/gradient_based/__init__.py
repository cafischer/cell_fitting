from collections import Iterable
import copy
import numpy as np

__author__ = 'caro'


class LineSearchError(RuntimeError):
    pass


def numerical_gradient(x, f, h=1e-8, method='central', *args, **kwargs):
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

    return gradient


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


def conjugate_gradient(fun, jac, x0, maxiter, c1=1e-4, c2=0.1, gtol=1e-5):
    x = [x0]
    f_values = np.zeros(maxiter)
    f_values[0] = fun(x0)
    df_values = np.zeros(maxiter, dtype=object)
    df_values[0] = jac(x0)
    p = np.zeros(maxiter, dtype=object)
    p[0] = -df_values[0]
    beta = np.zeros(maxiter)
    gnorm = gtol + 1

    for k in range(maxiter-1):
        if (gnorm <= gtol):
            break
        try:
            if k == 0:
                alpha = linesearch(fun, jac, x[k], p[k], c1, c2, None, maxiter=20)
            else:
                alpha = linesearch(fun, jac, x[k], p[k], c1, c2, f_values[k - 1], maxiter=20)
        except LineSearchError:
            print 'Line search failed!'
            #break
            alpha = 1
        x.append(x[k] + alpha * p[k])
        f_values[k+1] = fun(x[k+1])
        df_values[k+1] = jac(x[k+1])
        beta[k+1] = np.dot((df_values[k+1]-df_values[k]), df_values[k+1]) / vecnorm(df_values[k])
        beta[k+1] = max(0, beta[k+1])
        p[k+1] = -df_values[k+1] + beta[k+1] * p[k]
        gnorm = vecnorm(df_values[k+1])
    return x


def vecnorm(x, order=2):
    return np.sum(np.abs(x) ** order, axis=0) ** (1.0 / order)


def linesearch(fun, jac, x, p, c1, c2, old_phi0, maxiter):
    """
    Find step length alpha that satisfies the strong Wolfe conditions.
    :param fun:
    :type fun:
    :param jac:
    :type jac:
    :param x:
    :type x:
    :param p:
    :type p:
    :param a_max:
    :type a_max:
    :param c1:
    :type c1:
    :param c2:
    :type c2:
    :param old_phi0: fun(x_k-1)
    :type old_phi0:
    :return:
    :rtype:
    """
    phi = get_phi(fun, x, p)
    dphi = get_dphi(jac, x, p)
    phi0 = phi(0)
    dphi0 = dphi(0)
    alpha0 = 0
    if old_phi0 is not None:
        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_phi0) / dphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0
    phi_a1 = phi(alpha1)
    phi_a0 = phi0
    dphi_a0 = dphi0

    for i in xrange(maxiter):
        if (phi_a1 > phi0 + c1 * alpha1 * dphi0) or ((phi_a1 >= phi_a0) and (i > 0)):
            alpha_star = zoom(alpha0, alpha1, phi_a0, phi_a1, dphi_a0, phi0, dphi0, phi, dphi, c1, c2)
            break
        dphi_a1 = dphi(alpha1)
        if np.abs(dphi_a1) <= -c2 * dphi0:
            alpha_star = alpha1
            break
        if dphi_a1 >= 0:
            alpha_star = zoom(alpha1, alpha0, phi_a1, phi_a0, dphi_a1, phi0, dphi0, phi, dphi, c1, c2)
            break
        alpha0 = alpha1
        alpha1 = 2 * alpha1
        phi_a0 = phi_a1
        dphi_a0 = dphi_a1
        phi_a1 = phi(alpha1)
    if alpha_star is None:
        raise LineSearchError
    return alpha_star


def get_phi(fun, x, p):
    def phi(a):
        return fun(x + a * p)
    return phi


def get_dphi(jac, x, p):
    def dphi(a):
        return np.dot(jac(x + a * p), p)
    return dphi


def zoom(alo, ahi, phi_lo, phi_hi, dphi_lo, phi0, dphi0, phi, dphi, c1, c2):
    maxiter = 20
    delta1 = 0.2
    delta2 = 0.1

    for i in xrange(maxiter):
        dalpha = ahi - alo
        if dalpha < 0:
            a, b = ahi, alo
        else:
            a, b = alo, ahi

        if i > 0:
            margin_cubicmin = delta1 * dalpha
            aj = cubicmin(alo, phi_lo, dphi_lo, ahi, phi_hi, a_stored, phi_a_stored)
        if (i == 0) or (aj is None) or (aj > b - margin_cubicmin) or (aj < a + margin_cubicmin):
            margin_quadmin = delta2 * dalpha
            aj = quadmin(alo, phi_lo, dphi_lo, ahi, phi_hi)
            if (aj is None) or (aj > b - margin_quadmin) or (aj < a + margin_quadmin):
                aj = alo + 0.5 * dalpha

        phi_aj = phi(aj)
        if phi_aj > phi0 + c1 * aj * dphi0 or phi_aj >= phi(alo):
            a_stored = ahi
            phi_a_stored = phi_hi
            ahi = aj
            phi_hi = phi_aj
        else:
            dphi_aj = dphi(aj)
            if np.abs(dphi_aj) <= -c2 * dphi0:
                a_star = aj
                break
            if dphi_aj*(ahi-alo) >= 0:
                a_stored = ahi
                phi_a_stored = phi_hi
                ahi = alo
                phi_hi = phi_lo
            else:
                a_stored = alo
                phi_a_stored = phi_lo
            alo = aj
            phi_lo = phi_aj
            dphi_lo = dphi_aj

    else:
        a_star = None
    return a_star


def quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa,
    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    D = fa
    C = fpa
    diff = b - a * 1.0
    if diff == 0:
        return None
    B = (fb - D - C * diff) / (diff * diff)
    if B <= 0:
        return None
    xmin = a - C / (2.0 * B)
    return xmin


def cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    If no minimizer can be found return None
    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    C = fpa
    db = b - a
    dc = c - a
    if (db == 0) or (dc == 0) or (b == c):
        return None
    denom = (db * dc) ** 2 * (db - dc)
    d1 = np.empty((2, 2))
    d1[0, 0] = dc ** 2
    d1[0, 1] = -db ** 2
    d1[1, 0] = -dc ** 3
    d1[1, 1] = db ** 3
    [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                    fc - fa - C * dc]).flatten())
    A /= denom
    B /= denom
    radical = B * B - 3 * A * C
    if radical < 0:
        return None
    if A == 0:
        return None
    xmin = a + (-B + np.sqrt(radical)) / (3 * A)
    return xmin