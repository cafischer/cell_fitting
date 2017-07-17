import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell
from optimization.simulate import extract_simulation_params, simulate_gates, iclamp_handling_onset
from optimization.helpers import get_channel_list

# parameters
data_dir = '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv'
save_dir = '../../results/server/2017-07-06_13:50:52/434/L-BFGS-B/'
model_dir = os.path.join(save_dir, 'model', 'best_cell.json')
# model_dir = '../../results/server/2017-07-06_13:50:52/434/L-BFGS-B/model/best_cell.json'
mechanism_dir = '../../model/channels/vavoulis'

# load model
cell = Cell.from_modeldir(model_dir, mechanism_dir)

# simulate and record conductance
data = pd.read_csv(data_dir)
sim_params = extract_simulation_params(data)
v_model, t_model, i_model = iclamp_handling_onset(cell, **sim_params)
gates = simulate_gates(cell, sim_params)

# get gates
channel_list = get_channel_list(cell, 'soma')
gate_names = [gate for gate in gates.keys()]

# get input resistance
g_per_channel = np.zeros((len(channel_list), len(data.t)))
for i, channel in enumerate(channel_list):
    if channel == 'pas':
        g_per_channel[i, :] = np.ones(len(data.t)) * cell.soma(.5).g_pas
    for gate_name in gate_names:
        if channel in gate_name:
            g_per_channel[i, :] += gates[gate_name] * getattr(cell.soma(.5), channel).gbar
sum_g = np.sum(g_per_channel, 0)

v_model_scaled = v_model / (np.max(v_model)-np.min(v_model)) * (np.max(sum_g) - np.min(sum_g))
v_model_shift = v_model_scaled - v_model_scaled[0] + sum_g[0]

# plot
pl.figure()
pl.plot(t_model, v_model_shift, 'k', label='Vm (scaled)')
pl.plot(t_model, sum_g, 'b', label='$\sum_{channel} g_{channel}$')
for i, g in enumerate(g_per_channel):
    pl.plot(t_model, g, label=channel_list[i])
pl.legend()
pl.show()