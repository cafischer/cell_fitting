import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import json
from new_optimization.evaluation.evaluate import *
from new_optimization.fitter import *


if __name__ == '__main__':
    save_dir = '../../results/new_optimization/2015_08_06d/15_02_17_PP(4)/L-BFGS-B/'
    data_dir = '../../data/2015_08_06d/raw/rampIV/3.5(nA).csv'

    # load data
    data = pd.read_csv(data_dir)
    v_exp = np.array(data.v)
    i_exp = np.array(data.i)
    t_exp = np.array(data.t)
    dt = t_exp[1] - t_exp[0]
    dvdt_exp = np.concatenate((np.array([(v_exp[1]-v_exp[0])/dt]), np.diff(v_exp) / dt))

    # compute dvdt model
    data = pd.read_csv(data_dir)
    simulation_params = extract_simulation_params(data)
    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    best_candidate = get_best_candidate(save_dir, n_best=1)
    fitter.update_cell(best_candidate)

    v_model, _, _ = iclamp_handling_onset(fitter.cell, **simulation_params)
    dvdt_model = np.concatenate((np.array([(v_model[1]-v_model[0])/dt]), np.diff(v_model) / dt))

    # plot
    fig, (ax1, ax2) = pl.subplots(1, 2)
    ax1.plot(v_exp, dvdt_exp, 'k', label='Data')
    ax1.plot(v_exp, dvdt_model, 'r', label='Model')
    ax1.set_xlabel('V (mV)', fontsize=16)
    ax1.set_ylabel('dV/dt (mV/ms)', fontsize=16)
    pl.legend()
    ax2.plot(t_exp, v_exp, 'k', label='Data')
    ax2.plot(t_exp, v_model, 'r', label='Model')
    ax2.set_xlabel('Time (ms)', fontsize=16)
    ax2.set_ylabel('V (mV)', fontsize=16)
    pl.legend(fontsize=16)
    pl.tight_layout()
    pl.show()