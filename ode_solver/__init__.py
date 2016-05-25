import numpy as np
from scipy.optimize import newton

__author__ = 'caro'


def implicit_euler(x_now, x_old, t_now, t_old, f, *args):
    """
    Defintion of implicit Euler solving for the variable x_now.
    :param x_now: Variable for which the ode shall be solved.
    :type x_now: float
    :param x_old: Value of the variable in the previous time step.
    :type x_old: float
    :param t_now: Time of the current time step.
    :type t_now: float
    :param t_old: Time of the previous time step.
    :type t_old: float
    :param f: Derivative (ode) of the variable to be solved.
    :type f: function
    :param args: Additional arguments for the derivative (ode).
    :type args: tuple
    :return: Implicit Euler definition solving for x_now.
    :rtype: float
    """
    return x_now - x_old - (t_now-t_old) * f(x_now, t_now, *args)


def solve_implicit_euler(x0, t, f, x_guess, *args):
    """
    Numerically solves the ode defined in f with the implicit Euler method.
    :param x0: Initial value of the variable to be solved.
    :type x0: float
    :param t: Array of the time steps at which the ode shall be solved.
    :type t: array
    :param f: Derivative (ode) of the variable to be solved.
    :type f: function
    :param x_guess: Guess of the variable to be solved based on the value from the previous time step.
    :type x_guess: function
    :param args: Additional arguments for the derivative (ode).
    :type args: tuple
    :return: Variable integrated for each time step with implicit Euler.
    :rtype: array
    """
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = newton(implicit_euler, x_guess(x[i-1]), args=(x[i-1], t[i], t[i-1], f) + args)
    return x