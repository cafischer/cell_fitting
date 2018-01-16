import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as pl
pl.style.use('paper')


def get_spike_characteristics_dict():
    spike_characteristics_dict = {
        'AP_threshold': -10,  # mV
        'AP_interval': 2.5,  # ms
        'fAHP_interval': 4.0,
        'AP_width_before_onset': 2.0,  # ms
        'DAP_interval': 10.0,  # ms
        'order_fAHP_min': 1.0,  # ms (how many points to consider for the minimum)
        'order_DAP_max': 1.0,  # ms (how many points to consider for the minimum)
        'min_dist_to_DAP_max': 0.5,  # ms
        'k_splines': 3,
        's_splines': 0  # 0 means no interpolation, use for models
    }
    return spike_characteristics_dict


def get_characteristic_name_dict():
    return {'DAP_amp': 'DAP Amplitude', 'DAP_deflection': 'DAP Deflection', 'DAP_width': 'DAP Width',
            'DAP_time': 'DAP Time', 'fAHP_amp': 'fAHP Amplitude'}


def get_characteristic_unit_dict():
    return {'DAP_amp': 'mV', 'DAP_deflection': 'mV', 'DAP_width': 'ms', 'DAP_time': 'ms', 'fAHP_amp': 'mV'}


def plot_v(t, v, c='r', save_dir_img=None):
    fig = pl.figure()
    pl.plot(t, v, c, label='Model')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.tight_layout()
    if save_dir_img is not None:
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)
        pl.savefig(os.path.join(save_dir_img, 'v.png'))
    return fig


def joint_plot_data_and_model(x_data, y_data, x_model, y_model, x_name, y_name, save_dir_img=None):
    data = pd.DataFrame(np.array([x_data, y_data]).T, columns=[x_name, y_name])
    jp = sns.jointplot(x_name,y_name, data=data, stat_func=None, color='k', alpha=0.5)
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = x_model
    jp.y = y_model
    jp.plot_joint(pl.scatter, c='r', alpha=0.5)
    if save_dir_img is not None:
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)
        pl.savefig(os.path.join(save_dir_img, x_name+'_'+y_name+'.png'))
    return jp.fig