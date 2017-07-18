import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import os
from new_optimization.fitter import extract_simulation_params
from cell_characteristics.fIcurve import compute_fIcurve, compute_fIcurve_last_ISI
from util import merge_dicts
from optimization.simulate import iclamp_adaptive_handling_onset
from nrn_wrapper import Cell


if __name__ == '__main__':

    # parameters
    data_dir = '../../data/2015_08_26b/vrest-75/IV/'
    save_dir = '../../results/server/2017-07-17_17:05:19/54/L-BFGS-B/'
    model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../results/hand_tuning/cell434_5/'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # fI-curve for data
    v_traces_data = list()
    i_traces_data = list()
    for file_name in os.listdir(data_dir):
        data = pd.read_csv(data_dir+file_name)
        v_traces_data.append(data.v.values)
        i_traces_data.append(data.i.values)
    t_trace = data.t.values
    amps, firing_rates_data = compute_fIcurve(v_traces_data, i_traces_data, t_trace)
    _, firing_rates_data_last_ISI  = compute_fIcurve_last_ISI(v_traces_data, i_traces_data, t_trace)

    # discontinuities for IV
    dt = 0.05
    start_step = int(round(250 / dt))
    end_step = int(round(750 / dt))
    discontinuities_IV = [start_step, end_step]

    # fI-curve for model
    sim_params = {'celsius': 35, 'onset': 200, 'atol': 1e-6, 'continuous': True, 'discontinuities': discontinuities_IV,
                  'interpolate': True}
    sim_params = {'celsius': 35, 'onset': 200}
    v_traces_model = list()
    for file_name in os.listdir(data_dir):
        data = pd.read_csv(data_dir+file_name)
        simulation_params = merge_dicts(extract_simulation_params(data), sim_params)
        v_model, t_model, _ = iclamp_adaptive_handling_onset(cell, **simulation_params)
        v_traces_model.append(v_model)

    amps, firing_rates_model = compute_fIcurve(v_traces_model, i_traces_data, t_trace)
    _, firing_rates_model_last_ISI = compute_fIcurve_last_ISI(v_traces_model, i_traces_data, t_trace)

    # sort according to amplitudes
    idx_sort = np.argsort(amps)
    amps = amps[idx_sort]
    firing_rates_data = firing_rates_data[idx_sort]
    firing_rates_model = firing_rates_model[idx_sort]
    firing_rates_data_last_ISI = firing_rates_data_last_ISI[idx_sort]
    firing_rates_model_last_ISI = firing_rates_model_last_ISI[idx_sort]
    v_traces_model = np.array(v_traces_model)[idx_sort]

    # only take amps >= 0
    amps_greater0 = amps >= 0
    amps = amps[amps_greater0]
    firing_rates_data = firing_rates_data[amps_greater0]
    firing_rates_model = firing_rates_model[amps_greater0]
    firing_rates_data_last_ISI = firing_rates_data_last_ISI[amps_greater0]
    firing_rates_model_last_ISI = firing_rates_model_last_ISI[amps_greater0]
    v_traces_model = v_traces_model[amps_greater0]

    # plot
    save_dir_fig = os.path.join(save_dir, 'img/IV')
    if not os.path.exists(save_dir_fig):
        os.makedirs(save_dir_fig)

    pl.figure()
    pl.plot(amps, firing_rates_data, 'k', label='Exp. Data')
    pl.plot(amps, firing_rates_model, 'r', label='Model')
    pl.xlabel('Current (nA)', fontsize=16)
    pl.ylabel('Firing rate (APs/ms)', fontsize=16)
    pl.legend(loc='lower right', fontsize=16)
    pl.savefig(os.path.join(save_dir_fig, 'fIcurve.png'))
    pl.show()

    pl.figure()
    pl.plot(amps, firing_rates_data_last_ISI, 'k', label='Exp. Data')
    pl.plot(amps, firing_rates_model_last_ISI, 'r', label='Model')
    pl.xlabel('Current (nA)', fontsize=16)
    pl.ylabel('last ISI (ms)', fontsize=16)
    pl.legend(loc='lower right', fontsize=16)
    pl.savefig(os.path.join(save_dir_fig, 'fIcurve_last_ISI.png'))
    pl.show()

    for v_trace in v_traces_model:
        pl.figure()
        pl.plot(t_model, v_trace)
        pl.show()