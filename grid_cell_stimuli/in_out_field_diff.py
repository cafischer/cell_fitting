from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os


if __name__ == '__main__':

    save_dir = './results/test0/in_out_field_diff'
    save_dir_ramp_and_theta = './results/test0/ramp_and_theta'
    save_dir_in_out_fields = './results/test0/in_out_fields'

    # load
    ramp = np.load(os.path.join(save_dir_ramp_and_theta, 'ramp.npy'))
    theta_envelope = np.load(os.path.join(save_dir_ramp_and_theta, 'theta_envelope.npy'))
    t = np.load(os.path.join(save_dir_ramp_and_theta, 't.npy'))
    dt = t[1] - t[0]
    in_field = np.load(os.path.join(save_dir_in_out_fields, 'in_field.npy'))
    out_field = np.load(os.path.join(save_dir_in_out_fields, 'out_field.npy'))

    # in and out field difference
    ramp_in_mean = np.mean(ramp[in_field])
    ramp_out_mean = np.mean(ramp[out_field])
    theta_in_mean = np.mean(theta_envelope[in_field])
    theta_out_mean = np.mean(theta_envelope[out_field])
    ramp_diff = ramp_in_mean - ramp_out_mean
    theta_diff = theta_in_mean - theta_out_mean

    # save and plots
    print('$\Delta$Ramp: ', ramp_diff)  # exp: 2.9 +- 0.3 mV
    print('$\Delta$Theta: ', theta_diff)  # exp:  0.72 +- 0.12 mV