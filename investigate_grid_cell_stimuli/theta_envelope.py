import numpy as np
import os
from grid_cell_stimuli.theta_envelope import compute_envelope, plot_envelope


if __name__ == '__main__':
    folder = 'test0'
    save_dir = './results/' + folder + '/ramp_and_theta'
    save_dir_data = './results/' + folder + '/ramp_and_theta'

    # load
    theta = np.load(os.path.join(save_dir_data, 'theta.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    dt = t[1] - t[0]

    # Hilbert transform
    theta_envelope = compute_envelope(theta)

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'theta_envelope.npy'), theta_envelope)

    plot_envelope(theta, theta_envelope, t, save_dir)