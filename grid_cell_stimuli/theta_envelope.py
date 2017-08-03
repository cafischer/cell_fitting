import numpy as np
import matplotlib.pyplot as pl
import os
from scipy.signal import hilbert


if __name__ == '__main__':
    save_dir = './results/test0/ramp_and_theta'
    save_dir_data = './results/test0/ramp_and_theta'

    # load
    theta = np.load(os.path.join(save_dir_data, 'theta.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    dt = t[1] - t[0]

    # Hilbert transform
    theta_envelope = np.abs(hilbert(theta))

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'theta_envelope.npy'), theta_envelope)

    pl.figure()
    pl.plot(t, theta, 'b', label='Theta')
    pl.plot(t, theta_envelope, 'r', label='Theta')
    pl.ylabel('Voltage (mV)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.savefig(os.path.join(save_dir, 'theta_envelope.png'))
    pl.show()