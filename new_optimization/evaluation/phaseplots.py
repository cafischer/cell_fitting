import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import json
from new_optimization.evaluation.evaluate import FitterFactory, get_best_candidate, get_candidate_params
from optimization.simulate import iclamp_handling_onset, extract_simulation_params


if __name__ == '__main__':
    save_dir = '../../results/new_optimization/2015_08_06d/27_03_17_readjust/L-BFGS-B/'
    data_dir = '../../data/2015_08_06d/correct_vrest_-16mV/rampIV/3.5(nA).csv'
    n_best = 0

    # load data
    data = pd.read_csv(data_dir)
    v_exp = np.array(data.v)
    i_exp = np.array(data.i)
    t_exp = np.array(data.t)
    dt = t_exp[1] - t_exp[0]
    dvdt_exp = np.concatenate((np.array([(v_exp[1]-v_exp[0])/dt]), np.diff(v_exp) / dt))

    # get candidate
    data = pd.read_csv(data_dir)
    simulation_params = extract_simulation_params(data)
    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    best_candidate = get_candidate_params(get_best_candidate(save_dir, n_best=n_best))
    fitter.update_cell(best_candidate)

    # record v and channel gates
    mechanisms = fitter.cell.get_dict()['soma']['mechanisms'].keys()
    ion_channels = np.array(mechanisms)[np.array(['_ion' not in m for m in mechanisms])]
    gates = {}
    for ion_channel in ion_channels:
        gate_names = []
        for gate_name in ['m', 'n', 'h', 'l']:
            if getattr(getattr(fitter.cell.soma(.5), ion_channel), gate_name, None) is not None:
                gate_names.append(gate_name)
        for gate_name in gate_names:
            gates[ion_channel+'_'+gate_name] = fitter.cell.soma.record_from(ion_channel, gate_name)
    v_model, _, _ = iclamp_handling_onset(fitter.cell, **simulation_params)
    dvdt_model = np.concatenate((np.array([(v_model[1]-v_model[0])/dt]), np.diff(v_model) / dt))

    # plot
    for gate_name, gate_trace in gates.iteritems():
        gate_trace = np.array(gate_trace)[int(simulation_params['onset']/simulation_params['dt']):]
        dgatedt = np.concatenate((np.array([(gate_trace[1]-gate_trace[0])/dt]), np.diff(gate_trace) / dt))
        fig, (ax1, ax2) = pl.subplots(1, 2)
        ax1.plot(v_model, gate_trace, 'r')
        ax1.set_xlabel('V (mV)', fontsize=16)
        ax1.set_ylabel(gate_name, fontsize=16)
        ax2.plot(v_exp, dvdt_exp, 'k', label='Data')
        ax2.plot(v_model, dvdt_model, 'r', label='Model')
        ax2.set_xlabel('V (mV)', fontsize=16)
        ax2.set_ylabel('dV/dt (mV/ms)', fontsize=16)
        pl.legend(fontsize=16)
        pl.tight_layout()
        pl.show()