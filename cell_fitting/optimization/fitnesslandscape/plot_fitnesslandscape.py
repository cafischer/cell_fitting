import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from optimization.fitnesslandscape import *


def plot_fitnesslandscape(save_dir, title, error, has_1AP_mat, p1_range, p2_range, optimum, minima_xy):

    P1, P2 = np.meshgrid(p1_range, p2_range)
    fig, ax = pl.subplots()
    im = ax.pcolormesh(P1, P2, np.ma.masked_invalid(error).T)
    for minimum in minima_xy:
        if has_1AP_mat[minimum[0], minimum[1]]:
            ax.plot(p1_range[minimum[0]], p2_range[minimum[1]], 'ow')
        else:
            ax.plot(p1_range[minimum[0]], p2_range[minimum[1]], 'ow')
    ax.plot(optimum[0], optimum[1], 'x', color='k', mew=2, ms=8)
    pl.xlabel('$g_{na}$', fontsize=15)
    pl.ylabel('$g_{k}$', fontsize=15)
    pl.xlim(p1_range[0], p1_range[-1])
    pl.ylim(p2_range[0], p2_range[-1])
    pl.title(title, fontsize=15)
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Fitness', fontsize=15)
    pl.savefig(save_dir)
    pl.show()


def plot_fitnesslandscape3d(save_dir, error, has_1AP_mat, p1_range, p2_range, optimum, minima_xy):

    P1, P2 = np.meshgrid(p1_range, p2_range)
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    im = ax.plot_surface(P1, P2, np.ma.masked_invalid(error).T, cmap='Greys', rstride=1, cstride=1)
    for minimum in minima_xy:
        if has_1AP_mat[minimum[0], minimum[1]]:
            ax.plot([p1_range[minimum[0]]], [p2_range[minimum[1]]], [error[minimum[0], minimum[1]]], 'ow')
        else:
            ax.plot([p1_range[minimum[0]]], [p2_range[minimum[1]]], [error[minimum[0], minimum[1]]], 'ok')
    ax.plot([optimum[0]], [optimum[1]], [0], 'x', color='k', mew=2, ms=8)
    pl.xlabel('gmax na')
    pl.ylabel('gmax k')
    pl.title(fitfun)
    fig.colorbar(im)
    pl.savefig(save_dir)
    pl.show()


def plot_fitnesslandscape_with_minima_descent(save_dir, title, error, has_1AP_mat, p1_range, p2_range, optimum,
                                              minima_xy, minima_descent):

    P1, P2 = np.meshgrid(p1_range, p2_range)
    fig, ax = pl.subplots()
    im = ax.pcolormesh(P1, P2, np.ma.masked_invalid(error).T)
    for minimum in minima_xy:
        if has_1AP_mat[minimum[0], minimum[1]]:
            ax.plot(p1_range[minimum[0]], p2_range[minimum[1]], 'ow')
        else:
            ax.plot(p1_range[minimum[0]], p2_range[minimum[1]], 'ok')
    for minimum in minima_descent:
        ax.plot(minimum[0], minimum[1], 'o', color='0.5', markersize=5)
    ax.plot(optimum[0], optimum[1], 'x', color='k', mew=2, ms=8)
    pl.xlabel('gmax na')
    pl.ylabel('gmax k')
    pl.xlim(p1_range[0], p1_range[-1])
    pl.ylim(p2_range[0], p2_range[-1])
    pl.title(title)
    fig.colorbar(im)
    pl.savefig(save_dir)
    pl.show()


def plot_1AP(save_dir, has_1AP_mat, p1_range, p2_range):

    P1, P2 = np.meshgrid(p1_range, p2_range)
    fig, ax = pl.subplots()
    im = ax.pcolormesh(P1, P2, np.ma.masked_invalid(has_1AP_mat).T, cmap='Greys')
    ax.plot(optimum[0], optimum[1], 'x', color='0.5', mew=2, ms=8)
    pl.xlabel('gmax na')
    pl.ylabel('gmax k')
    pl.title('Models with 1 AP')
    pl.savefig(save_dir)
    pl.show()


def bar_plot_minima(save_dir, fitfuns):
    len_minima = np.zeros(len(fitfuns))
    for i, fitfun in enumerate(fitfuns):
        with open(save_dir + fitfun + '/minima2d.npy', 'r') as f:
            minima = np.load(f)
            len_minima[i] = len(minima)
    width = 0.5
    pl.figure()
    pl.bar(np.arange(len(fitfuns)), len_minima, width, color='0.5')
    pl.xticks(np.arange(len(fitfuns)) + width/2, fitfuns)
    pl.ylabel('# Minima')
    pl.show()



if __name__ == '__main__':
    #save_dir = '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/fitfuns/'
    #fitfuns = ['v_rest', 'APamp+v_rest', 'APamp+v_trace', 'v_trace+v_rest']
    #['v_trace', 'v_rest', 'phasehist', 'APamp', 'APwidth', 'APtime', 'APshift']
    #bar_plot_minima(save_dir, fitfuns)

    save_dir = '../../results/fitnesslandscapes/modellandscape/gna_gk/'
    #method = 'CG'
    #save_dir_minima = '../../results/fitnesslandscapes/find_local_minima/performance/gna_gk/whole_region/phasehist/' + method + '/'
    new_folder = 'fitfuns/v_trace'
    fitfun = 'v_trace'
    order = 1
    optimum = [0.12, 0.036]

    with open(save_dir + '/' + new_folder + '/error.npy', 'r') as f:
        error = np.load(f)
    with open(save_dir + '/' + new_folder + '/has_1AP.npy', 'r') as f:
        has_1AP_mat = np.load(f)
    p1_range = np.loadtxt(save_dir + '/p1_range.txt')
    p2_range = np.loadtxt(save_dir + '/p2_range.txt')

    #save_dir_tmp = save_dir + '1AP.png'
    #plot_1AP(save_dir_tmp, has_1AP_mat, p1_range, p2_range)

    minima_xy = get_local_minima_2d(error)
    with open(save_dir + '/' + new_folder + '/minima2d.npy', 'w') as f:
        np.save(f, np.array(minima_xy))
    save_dir_tmp = save_dir + '/' + new_folder + '/fitness_landscape_' + str(order) + '.png'
    plot_fitnesslandscape(save_dir_tmp, fitfun, error, has_1AP_mat, p1_range, p2_range, optimum, minima_xy)

    #save_dir_tmp = save_dir + '/' + new_folder + '/fitness_landscape3d_' + str(order) + '.png'
    #plot_fitnesslandscape3d(save_dir_tmp, error, has_1AP_mat, p1_range, p2_range, optimum, minima_xy)

    #with open(save_dir_minima + '/minima_descent.npy', 'r') as f:
    #    minima_descent = np.load(f)
    #save_dir_tmp = save_dir + '/' + new_folder + '/fitness_landscape_descent_' + method + '.png'
    #plot_fitnesslandscape_with_minima_descent(save_dir_tmp, error, has_1AP_mat, p1_range, p2_range, optimum,
    #                                          minima_xy, minima_descent)

    #with open(save_dir_minima + '/minima_success.npy', 'r') as f:
    #    minima_success = np.load(f)
    #save_dir_tmp = save_dir + '/' + new_folder + '/fitness_landscape_success_' + method + '.png'
    #plot_fitnesslandscape_with_minima_descent(save_dir_tmp, error, has_1AP_mat, p1_range, p2_range, optimum,
    #                                          minima_xy, minima_success)