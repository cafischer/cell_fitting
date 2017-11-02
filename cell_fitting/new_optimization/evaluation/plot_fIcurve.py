import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.new_optimization.fitter import extract_simulation_params
from cell_characteristics.fIcurve import compute_fIcurve, compute_fIcurve_last_ISI
from cell_fitting.util import merge_dicts
from cell_fitting.optimization.simulate import iclamp_adaptive_handling_onset
from nrn_wrapper import Cell
from cell_characteristics.analyze_APs import get_AP_max_idx, get_AP_start_end
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    data_dir = '../../data/2015_08_26b/vrest-75/IV/'

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
    v_traces_data = np.array(v_traces_data)[idx_sort]
    v_traces_model = np.array(v_traces_model)[idx_sort]

    # only take amps >= 0
    amps_greater0_idx = amps >= 0
    amps_greater0 = amps[amps_greater0_idx]
    firing_rates_data = firing_rates_data[amps_greater0_idx]
    firing_rates_model = firing_rates_model[amps_greater0_idx]
    firing_rates_data_last_ISI = firing_rates_data_last_ISI[amps_greater0_idx]
    firing_rates_model_last_ISI = firing_rates_model_last_ISI[amps_greater0_idx]
    #v_traces_data = v_traces_data[amps_greater0]
    #v_traces_model = v_traces_model[amps_greater0]

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'IV')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    pl.figure()
    #pl.plot(amps_greater0, firing_rates_data, '-ok', label='Exp. Data')
    pl.plot(amps_greater0, firing_rates_model, '-or', label='Model')
    pl.ylim([0, 0.09])
    pl.xlabel('Current (nA)')
    pl.ylabel('Firing rate (APs/ms)')
    #pl.legend(loc='lower right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'fIcurve.png'))
    #pl.show()

    pl.figure()
    pl.plot(amps_greater0, firing_rates_data_last_ISI, '-ok', label='Exp. Data')
    pl.plot(amps_greater0, firing_rates_model_last_ISI, '-or', label='Model')
    pl.xlabel('Current (nA)')
    pl.ylabel('last ISI (ms)')
    pl.legend(loc='upper right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'fIcurve_last_ISI.png'))
    #pl.show()

    for amp, v_trace_data, v_trace_model in zip(amps, v_traces_data, v_traces_model):
        AP_start_model, AP_end_model = get_AP_start_end(v_trace_model, threshold=-30, n=0)
        AP_start_data, AP_end_data = get_AP_start_end(v_trace_data, threshold=-30, n=0)

        pl.figure()
        if AP_start_model is not None and AP_start_data is not None:
            AP_peak_model = v_trace_model[get_AP_max_idx(v_trace_model, AP_start_model, AP_end_model)]
            AP_peak_data = v_trace_data[get_AP_max_idx(v_trace_data, AP_start_data, AP_end_data)]
            if AP_peak_model > AP_peak_data:
                pl.plot(t_model, v_trace_model, 'r', label='Model')
                #pl.plot(t_trace, v_trace_data, 'k', label='Exp. Data')
            else:
                #pl.plot(t_trace, v_trace_data, 'k', label='Exp. Data')
                pl.plot(t_model, v_trace_model, 'r', label='Model')
        else:
            #pl.plot(t_trace, v_trace_data, 'k', label='Exp. Data')
            pl.plot(t_model, v_trace_model, 'r', label='Model')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        if amp == -0.1:
            pl.ylim(-80, -60)
        #pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'IV' + str(amp) + '.png'))
        #pl.show()