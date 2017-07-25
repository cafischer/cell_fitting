from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from bac_project.connectivity.connection import synaptic_input
from nrn_wrapper import Cell, load_mechanism_dir
from optimization.simulate import iclamp_adaptive_handling_onset
from neuron import h
import os

load_mechanism_dir("/home/cf/Phd/programming/projects/bac_project/bac_project/connectivity/vecstim")

save_dir = os.path.join('../results/server/2017-07-06_13:50:52/434/L-BFGS-B', 'img', 'ISI_hist')
model_dir = '../results/server/2017-07-06_13:50:52/434/L-BFGS-B/model/cell.json'
mechanism_dir = '../model/channels/vavoulis'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# synaptic input
pos = 0.5
onset = 200
tstop = 59800  # 60 sec
dt = 0.01
freq = {'AMPA': 1000, 'NMDA': 100, 'GABA': 8000}
seeds_stim = {'AMPA': 1, 'NMDA': 1, 'GABA': 1}
n_stim = {'AMPA': 1, 'NMDA': 1, 'GABA': 1}

cell = Cell.from_modeldir(model_dir, mechanism_dir)


params_stim = {'kind': 'poisson', 'onset': onset, 'tstop': tstop, 'dt': dt,
               'freq': freq['AMPA'], 'pos': pos, 'seed': seeds_stim['AMPA']}
params_syn = {'tau1': 0.25, 'tau2': 2.5, 'e': 0, 'pos': pos}
params_weight = {'kind': 'same', 'weight': 0.001}
params_AMPA = {'cell': cell, 'n_stim': n_stim['AMPA'], 'section': 'soma', 'params_stim': params_stim,
                     'params_syn': params_syn, 'params_weight': params_weight, 'delay': 0}

params_stim = {'kind': 'poisson', 'onset': onset, 'tstop': tstop, 'dt': dt,
               'freq': freq['NMDA'], 'pos': pos, 'seed': seeds_stim['NMDA']}
params_syn = {'tau1': 5, 'tau2': 150, 'e': 0, 'pos': pos}
params_weight = {'kind': 'same', 'weight': 0.0001}
params_NMDA = {'cell': cell, 'n_stim': n_stim['NMDA'], 'section': 'soma', 'params_stim': params_stim,
                     'params_syn': params_syn, 'params_weight': params_weight, 'delay': 0}

params_stim = {'kind': 'poisson', 'onset': onset, 'tstop': tstop, 'dt': dt,
               'freq': freq['GABA'], 'pos': pos, 'seed': seeds_stim['GABA']}
params_syn = {'tau1': 0.5, 'tau2': 5, 'e': -75, 'pos': pos}
params_weight = {'kind': 'same', 'weight': 0.0001}
params_GABA = {'cell': cell, 'n_stim': n_stim['GABA'], 'section': 'soma', 'params_stim': params_stim,
                     'params_syn': params_syn, 'params_weight': params_weight, 'delay': 0}

# inputs
syn_AMPA, stim_AMPA, con_AMPA, weights_AMPA, spiketimes_AMPA = synaptic_input(**params_AMPA)
syn_NMDA, stim_NMDA, con_NMDA, weights_NMDA, spiketimes_NMDA = synaptic_input(**params_NMDA)
syn_GABA, stim_GABA, con_GABA, weights_GABA, spiketimes_GABA = synaptic_input(**params_GABA)

# record currents
i_AMPA = []
for syn in syn_AMPA:
    i = h.Vector()
    i.record(syn._ref_i, dt)
    i_AMPA.append(i)
i_NMDA = []
for syn in syn_NMDA:
    i = h.Vector()
    i.record(syn._ref_i, dt)
    i_NMDA.append(i)
i_GABA = []
for syn in syn_GABA:
    i = h.Vector()
    i.record(syn._ref_i, dt)
    i_GABA.append(i)

# run simulation
simulation_params = {'sec': ['soma', None], 'onset': onset, 'tstop': tstop, 'dt': dt,
                     'i_inj': np.zeros(int(round(tstop/dt+1))), 'v_init': -75, 'continuous': True, 'interpolate': True}
v, t, i_inj = iclamp_adaptive_handling_onset(cell, **simulation_params)

# make arrays
i_AMPA = np.array([i.to_python() for i in i_AMPA])
i_NMDA = np.array([i.to_python() for i in i_NMDA])
i_GABA = np.array([i.to_python() for i in i_GABA])

# plot
pl.figure()
pl.plot(t, v)
#pl.show()

start = int(round(onset/dt))
pl.figure()
t_ = np.linspace(0, tstop, len(np.sum(i_AMPA, 0)[start:]))
pl.plot(t_, np.sum(i_AMPA, 0)[start:], 'b', label='AMPA')
pl.plot(t_, np.sum(i_NMDA, 0)[start:], 'g', label='NMDA')
pl.plot(t_, np.sum(i_GABA, 0)[start:], 'r', label='GABA')
pl.legend()
pl.show()

# check for doublets
from cell_characteristics.analyze_APs import get_AP_onsets
AP_onsets = get_AP_onsets(v, -30)

# ISI hist
bins = np.arange(0, 1000+10, 10)
ISIs = np.diff(AP_onsets * dt)
pl.figure()
pl.hist(ISIs, bins=bins)
pl.savefig(os.path.join(save_dir, 'ISI_hist.png'))
pl.show()

# percent doublet (ISI < 50 ms) and theta (170 ms <= ISI < 330 ms)
hist, bins = np.histogram(ISIs, bins=bins)
n_ISI = len(ISIs)
n_doublets = np.sum(hist[bins[:-1] < 50])
percent_doublets = n_doublets / n_ISI
print('percent doublets: ', percent_doublets)
n_theta = np.sum(hist[np.logical_and(170 <= bins[:-1], bins[:-1] < 330)])
percent_theta = n_theta / n_ISI
print('percent theta: ', percent_theta)