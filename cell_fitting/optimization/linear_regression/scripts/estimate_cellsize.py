import numpy as np
import matplotlib.pyplot as pl
from optimization.helpers import *
from optimization.linear_regression import linear_regression, plot_fit
from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter

__author__ = 'caro'

# parameter
save_dir = '../../../results/linear_regression/estimate_cellsize/'

# create fiter
variable_keys = [
                    [['soma', '0.5', 'pas', 'g']],
                    [['soma', '0.5', 'km', 'gbar']],
                    [['soma', '0.5', 'ih_fast', 'gbar']],
                    [['soma', '0.5', 'ih_slow', 'gbar']],
                    [['soma', '0.5', 'nap', 'gbar']],
                    [['soma', '0.5', 'kdr', 'gbar']],
                    [['soma', '0.5', 'kap', 'gbar']],
                    [['soma', '0.5', 'na8st', 'gbar']]
                 ]
errfun = 'rms'
fitfun = ['get_v']
fitnessweights = [1]
model_dir = '../../../model/cells/dapmodel0.json'
mechanism_dir = '../../../model/vclamp/schmidthieber'
data_dir = '../../../data/2015_08_26b/simulate_rampIV/3.0(nA).csv'

fitter = HodgkinHuxleyFitter(variable_keys, errfun, fitfun, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params={'celsius': 35})

# create cell
candidate = np.ones(len(variable_keys))  # gbars should be 1
fitter.update_cell(candidate)

# extract parameter
channel_list = get_channel_list(fitter.cell, 'soma')
ion_list = get_ionlist(channel_list)

v_exp = fitter.data.v.values
t_exp = fitter.data.t.values
i_exp = fitter.data.i.values
dt_exp = t_exp[1] - t_exp[0]

dvdt_exp = np.concatenate((np.array([(v_exp[1]-v_exp[0])/dt_exp]), np.diff(v_exp)/dt_exp))
celsius = fitter.simulation_params['celsius']

# convert units
dvdt_sc, i_inj_sc, _, _, cell_area = convert_units(fitter.cell.soma.L, fitter.cell.soma.diam,
                                                   fitter.cell.soma.cm, dvdt_exp,
                                                   i_exp, np.zeros(len(t_exp)))

# extract part of current injection
current_inj = np.nonzero(i_exp)[0]
idx_start = current_inj[0]
idx_end = current_inj[-1] / dt_exp
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
fitter.cell.soma.L = L
fitter.cell.soma.diam = diam
cell_area = fitter.cell.soma(.5).area() * 1e-8
print 'L: ' + str(L)
print 'diam: ' + str(diam)
print 'cell area: ' + str(cell_area * 1e8)
