from optimization.fitnesslandscape.plot_fitnesslandscape import *


def plot_fitness():
    save_dir = '../../results/fitnesslandscapes/modellandscape/gna_gk/'
    new_folder = 'fitfuns/stefans_fun'
    fitfun = 'stefans_fun'
    optimum = [0.12, 0.036]

    with open(save_dir + '/' + new_folder + '/error.npy', 'r') as f:
        error = np.load(f)
    with open(save_dir + '/' + new_folder + '/has_1AP.npy', 'r') as f:
        has_1AP_mat = np.load(f)
    p1_range = np.loadtxt(save_dir + '/p1_range.txt')
    p2_range = np.loadtxt(save_dir + '/p2_range.txt')
    with open(save_dir + '/' + new_folder + '/minima2d.npy', 'r') as f:
        minima_xy = np.load(f)

    save_dir_tmp = save_dir + '/' + new_folder + '/fitness_landscape.png'
    plot_fitnesslandscape(save_dir_tmp, fitfun, error, has_1AP_mat, p1_range, p2_range, optimum, minima_xy)


def plot_fitness_descent():
    save_dir = '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/'
    save_dir_minima = '../../results/fitnesslandscapes/find_local_minima/performance/gna_gk/whole_region/APamp/CG/'
    new_folder = 'fitfuns/APamp'
    fitfun = 'APamp'
    optimum = [0.12, 0.036]

    with open(save_dir + '/' + new_folder + '/error.npy', 'r') as f:
        error = np.load(f)
    with open(save_dir + '/' + new_folder + '/has_1AP.npy', 'r') as f:
        has_1AP_mat = np.load(f)
    p1_range = np.loadtxt(save_dir + '/p1_range.txt')
    p2_range = np.loadtxt(save_dir + '/p2_range.txt')
    with open(save_dir + '/' + new_folder + '/minima2d.npy', 'r') as f:
        minima2d = np.load(f)
    with open(save_dir_minima + '/minima_success.npy', 'r') as f:
        minima_success = np.load(f)

    save_dir_tmp = save_dir + '/' + new_folder + '/fitness_landscape.png'
    plot_fitnesslandscape_with_minima_descent(save_dir_tmp, fitfun, error, has_1AP_mat, p1_range, p2_range, optimum,
                                              minima2d, minima_success)


if __name__ == '__main__':
    plot_fitness()
    #plot_fitness_descent()