from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell
from optimization.simulate import extract_simulation_params, iclamp_handling_onset, simulate_currents
import pandas as pd


def run_on(data_dir, dt=None, v_init=None, i_inj=None, plot=False):
    # load data
    data = pd.read_csv(data_dir)
    sim_params = extract_simulation_params(data)
    if dt is not None:
        sim_params['dt'] = dt
    if v_init is not None:
        sim_params['v_init'] = v_init
    if i_inj is not None:
        sim_params['i_inj'] = i_inj

    # simulate
    v, t, i = iclamp_handling_onset(cell, **sim_params)

    if plot:
        pl.figure()
        pl.plot(data.t, data.v, 'k')
        pl.plot(t, v, 'r')
        pl.show()

    return v, t, i



# create Cell
cell = Cell.from_modeldir('../model/cells/nowacki_model.json', '../model/channels/nowacki')

# parameters
v_init = -75.33
dt = 0.01

# simulations

# rampIV
run_on('../data/2015_08_06d/correct_vrest_-16mV/simulate_rampIV/3.3(nA).csv', v_init=v_init, plot=True)

# visualize currents
data = pd.read_csv('../data/2015_08_06d/correct_vrest_-16mV/simulate_rampIV/3.3(nA).csv')
sim_params = extract_simulation_params(data)
sim_params['v_init'] = -75.33

currents = simulate_currents(cell, sim_params, plot=True)

pl.figure()
pl.plot(data.t[1:], np.diff(data.v) / np.max(np.diff(data.v)), 'k')
pl.plot(data.t, -1*np.sum(currents) / np.max(-1*np.sum(currents)), 'r')
pl.show()

# plot_IV
i_inj = np.zeros(int(round(1000/dt, 0)))
i_inj[int(round(210/dt, 0)):int(round(710/dt, 0))] = 0.1
#run_on('../data/2015_08_06d/correct_vrest_-16mV/plot_IV/0.1(nA).csv', dt, v_init, i_inj, plot=True)

# PP(0)(3)
run_on('../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(0)(3)/0(nA).csv', v_init=v_init, plot=True)

# visualize currents
data = pd.read_csv('../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(0)(3)/0(nA).csv')
sim_params = extract_simulation_params(data)
sim_params['v_init'] = -75.33

currents = simulate_currents(cell, sim_params, plot=True)

pl.figure()
pl.plot(data.t[1:], np.diff(data.v) / np.max(np.diff(data.v)), 'k')
pl.plot(data.t, -1*np.sum(currents) / np.max(-1*np.sum(currents)), 'r')
pl.show()

# PP(0)(21)
run_on('../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(0)(21)/0(nA).csv', v_init=v_init, plot=True)

# plot_IV
run_on('../data/2015_08_06d/correct_vrest_-16mV/plot_IV/-0.1(nA).csv', v_init=v_init, plot=True)
run_on('../data/2015_08_06d/correct_vrest_-16mV/plot_IV/0.1(nA).csv', v_init=v_init, plot=True)
run_on('../data/2015_08_06d/correct_vrest_-16mV/plot_IV/1.0(nA).csv', v_init=v_init, plot=True)