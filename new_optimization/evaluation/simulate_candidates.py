import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import json
from optimization.errfuns import rms
from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter

__author__ = 'caro'


save_dir = '../../results/new_optimization/V_18_10_16/'
method = 'L-BFGS-B'

candidates = pd.read_csv(save_dir + method + '/candidates.csv')
best_candidate = candidates.candidate[np.argmin(candidates.fitness)]
best_candidate = np.array([float(x) for x in best_candidate.split()])

with open(save_dir + 'optimization_settings.json', 'r') as f:
    optimization_settings = json.load(f)

fitter = HodgkinHuxleyFitter(**optimization_settings['fitter'])

v_model, t, i_inj = fitter.simulate_cell(best_candidate)

pl.figure()
pl.plot(t, fitter.data.v, 'k', label='data')
pl.plot(t, v_model, 'r', label='model')
pl.legend()
pl.show()