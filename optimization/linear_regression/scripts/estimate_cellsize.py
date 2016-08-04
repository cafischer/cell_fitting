import numpy as np
import matplotlib.pyplot as pl

from optimization.problems import CellFitProblem, get_channel_list, get_ionlist, convert_units
from optimization.linear_regression import linear_regression, plot_fit

__author__ = 'caro'

# parameter
save_dir = '../../../results/linear_regression/estimate_cellsize/'
n_models = 4
dt_fac = np.arange(0, 5, 1)

variables = [
            [0, 2.5, [['soma', '0.5', 'na8st', 'gbar']]]
            ]

params = {
          'name': 'CellFitProblem',
          'maximize': False,
          'normalize': True,
          'model_dir': '../../../model/cells/toymodel1.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../../data/toymodels/toymodel1/ramp.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

# create problem
problem = CellFitProblem(**params)

# create cell
candidate = np.ones(len(problem.path_variables))  # gbars should be 1
problem.update_cell(candidate)

# extract parameter
channel_list = get_channel_list(problem.cell, 'soma')
ion_list = get_ionlist(channel_list)

v_exp = problem.data.v.values
t_exp = problem.data.t.values
i_exp = problem.data.i.values
dt_exp = t_exp[1] - t_exp[0]

dvdt_exp = np.concatenate((np.array([(v_exp[1]-v_exp[0])/dt_exp]), np.diff(v_exp)/dt_exp))
celsius = problem.simulation_params['celsius']

# convert units
dvdt_sc, i_inj_sc, _, _, cell_area = convert_units(problem.cell.soma.L, problem.cell.soma.diam,
                                                   problem.cell.soma.cm, dvdt_exp,
                                                   i_exp, np.zeros(len(t_exp)))

# extract part of current injection
current_inj = np.nonzero(i_exp)[0]
idx_start = current_inj[0]
idx_end = current_inj[-1]
dvdt_cut = dvdt_sc[idx_start:idx_end]
t_cut = t_exp[idx_start:idx_end]
i_exp_cut = i_inj_sc[idx_start:idx_end]
currents_cut = np.zeros(len(t_cut))

# linear regression
weights, residual, y, X = linear_regression(dvdt_cut, i_exp_cut, currents_cut, i_pas=0, cell_area=cell_area)

# plots
pl.figure()
pl.plot(t_exp, dvdt_exp, 'k')
pl.plot(t_cut, dvdt_cut, 'r')
pl.title('Visualize cut of dV/dt')
pl.show()

plot_fit(y, X, weights, t_cut, [])

# transform into change in cell area
# 1) area_new = area_old * cm_fit
area_new = cell_area * weights[-1]
# 2) r=L = np.sqrt(area_new / (2*np.pi*1e-8))
r = np.sqrt(area_new / (2*np.pi*1e-8))
L = np.sqrt(area_new / (2*np.pi*1e-8))
# 3) diam = 2*r
diam = 2*r
problem.cell.soma.L = L
problem.cell.soma.diam = diam
cell_area = problem.cell.soma(.5).area() * 1e-8
print 'L: ' + str(L)
print 'diam: ' + str(diam)
print 'cell area: ' + str(cell_area * 1e8)