import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import os
from new_optimization.evaluation.evaluate import *
from new_optimization.fitter import *
from cell_characteristics.fIcurve import *


if __name__ == '__main__':
    save_dir = '../../results/new_optimization/2015_08_06d/15_02_17_PP(4)/L-BFGS-B/'
    data_dir = '../../data/2015_08_06d/raw/IV/'
    n_best = 0

    # data
    v_traces_data = list()
    i_traces_data = list()
    for file_name in os.listdir(data_dir):
        data = pd.read_csv(data_dir+file_name)
        v_traces_data.append(data.v.values)
        i_traces_data.append(data.i.values)
    t_trace = data.t.values
    amps, firing_rates_data = compute_fIcurve(v_traces_data, i_traces_data, t_trace)

    # model
    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    load_mechanism_dir(optimization_settings['fitter_params']['mechanism_dir'])
    optimization_settings['fitter_params']['mechanism_dir'] = None

    v_traces_model = list()
    for file_name in os.listdir(data_dir):
        data = pd.read_csv(data_dir+file_name)
        simulation_params = extract_simulation_params(data)
        best_candidate = get_best_candidate(save_dir, n_best)
        fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
        fitter.update_cell(best_candidate)
        v_model, _, _ = iclamp_handling_onset(fitter.cell, **simulation_params)
        v_traces_model.append(v_model)

    amps, firing_rates_model = compute_fIcurve(v_traces_model, i_traces_data, t_trace)

    pl.figure()
    pl.plot(amps, firing_rates_data, 'k', label='Data')
    pl.plot(amps, firing_rates_model, 'r', label='Model')
    pl.xlabel('Current (nA)', fontsize=16)
    pl.ylabel('Firing rate (APs/ms)', fontsize=16)
    pl.legend(loc='lower right', fontsize=16)
    pl.show()