import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import seaborn as sns
from cell_characteristics.fIcurve import compute_fIcurve
from nrn_wrapper import Cell, load_mechanism_dir
from scipy.optimize import curve_fit

from cell_fitting.optimization.fitter import extract_simulation_params
from cell_fitting.optimization.simulate import iclamp_adaptive_handling_onset
from cell_fitting.util import merge_dicts

pl.style.use('paper')


def square_root(x, a, b):
    sr = np.sqrt(a * (x - b))
    sr[np.isnan(sr)] = 0
    return sr

if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'
    mechanism_dir = '../../../model/channels/vavoulis'
    data_dir = '../../../data/2015_08_26b/vrest-75/IV/'
    save_dir_data = '../../../data/plots/fI_curve/rat/summary'
    model_ids = range(1, 7)
    load_mechanism_dir(mechanism_dir)

    # current traces from data
    i_traces_data = list()
    for file_name in os.listdir(data_dir):
        data = pd.read_csv(data_dir + file_name)
        i_traces_data.append(data.i.values)
    t_trace = data.t.values

    FI_a_models = []
    FI_b_models = []

    for model_id in model_ids:
        save_dir_model = save_dir + str(model_id)

        # load model
        model_dir = os.path.join(save_dir_model, 'cell.json')
        cell = Cell.from_modeldir(model_dir)

        # discontinuities for IV
        dt = 0.05
        start_step = int(round(250 / dt))
        end_step = int(round(750 / dt))
        discontinuities_IV = [start_step, end_step]

        # fI-curve for model
        #sim_params = {'celsius': 35, 'onset': 200, 'atol': 1e-6, 'continuous': True,
        #              'discontinuities': discontinuities_IV, 'interpolate': True}
        sim_params = {'celsius': 35, 'onset': 200}
        v_traces_model = list()
        for file_name in os.listdir(data_dir):
            data = pd.read_csv(data_dir+file_name)
            simulation_params = merge_dicts(extract_simulation_params(data), sim_params)
            v_model, t_model, _ = iclamp_adaptive_handling_onset(cell, **simulation_params)
            v_traces_model.append(v_model)

        amps, firing_rates_model = compute_fIcurve(v_traces_model, i_traces_data, t_trace)

        # sort according to amplitudes
        idx_sort = np.argsort(amps)
        amps = amps[idx_sort]
        firing_rates_model = firing_rates_model[idx_sort]
        v_traces_model = np.array(v_traces_model)[idx_sort]

        # only take amps >= 0
        amps_greater0_idx = amps >= 0
        amps_greater0 = amps[amps_greater0_idx]
        firing_rates_model = firing_rates_model[amps_greater0_idx]

        # fit square root to FI-curve
        b0 = amps_greater0[np.where(firing_rates_model > 0)[0][0]]
        p_opt, _ = curve_fit(square_root, amps_greater0, firing_rates_model, p0=[0.005, b0])
        print p_opt
        FI_a_models.append(p_opt[0])
        FI_b_models.append(p_opt[1])

        # plot
        save_dir_img_model = os.path.join(save_dir_model, 'img', 'IV', 'fi_curve')
        if not os.path.exists(save_dir_img_model):
            os.makedirs(save_dir_img_model)

        pl.figure()
        pl.plot(amps_greater0, firing_rates_model, '-or', label='Model')
        pl.plot(amps_greater0, square_root(amps_greater0, p_opt[0], p_opt[1]), 'b')
        pl.xlabel('Current (nA)')
        pl.ylabel('Firing rate (APs/ms)')
        # pl.legend(loc='lower right')
        pl.ylim(0, 0.09)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img_model, 'fIcurve_fit.png'))
        #pl.show()

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'IV', 'fi_curve')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    FI_a = list(np.load(os.path.join(save_dir_data, 'FI_a.npy')))
    FI_b = list(np.load(os.path.join(save_dir_data, 'FI_b.npy')))
    data = pd.DataFrame(np.array([FI_a, FI_b]).T, columns=['Scaling', 'Shift'])
    jp = sns.jointplot('Scaling', 'Shift', data=data, stat_func=None, color='k') #, xlim=(0, 0.025), ylim=(0, 0.8))
    jp.fig.set_size_inches(6.4, 4.8)
    jp.x = FI_a_models
    jp.y = FI_b_models
    #pl.scatter(np.array(FI_a_models), np.array(FI_b_models), c='r')
    jp.plot_joint(pl.scatter, c='r')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'scaling_shift_hist.png'))
    pl.show()