import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as pl
from nrn_wrapper import Cell
from optimization.simulate import iclamp_handling_onset, extract_simulation_params


if __name__ == '__main__':
    # parameters
    data_dir = '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv'
    save_dir = '../../results/server/2017-07-06_13:50:52/434/L-BFGS-B/'
    model_dir = os.path.join(save_dir, 'model', 'best_cell.json')
    #model_dir = '../../results/server/2017-07-06_13:50:52/434/L-BFGS-B/model/best_cell.json'
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

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

    # record v and channel gates
    mechanisms = cell.get_dict()['soma']['mechanisms'].keys()
    ion_channels = np.array(mechanisms)[np.array(['_ion' not in m for m in mechanisms])]
    gates = {}
    for ion_channel in ion_channels:
        gate_names = []
        for gate_name in ['m', 'n', 'h', 'l']:
            if getattr(getattr(cell.soma(.5), ion_channel), gate_name, None) is not None:
                gate_names.append(gate_name)
        for gate_name in gate_names:
            gates[ion_channel+'_'+gate_name] = cell.soma.record_from(ion_channel, gate_name)
    v_model, _, _ = iclamp_handling_onset(cell, **simulation_params)
    dvdt_model = np.concatenate((np.array([(v_model[1]-v_model[0])/dt]), np.diff(v_model) / dt))

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'phaseplot')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

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
        pl.savefig(os.path.join(save_dir_img, 'phaseplot_'+gate_name+'.png'))
        pl.show()