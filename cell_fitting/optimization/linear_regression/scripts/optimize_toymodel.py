import json
import os

from nrn_wrapper import Cell

from cell_fitting.optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter
from cell_fitting.optimization.helpers import *
from cell_fitting.optimization.linear_regression import *
from cell_fitting.optimization.simulate import currents_given_v

__author__ = 'caro'

# parameter
save_dir = '../../../results/linear_regression/test/'
with_cm = False

#variables = [
#    [0, 1.0, [['soma', '0.5', 'na_hh', 'gbar']]],
#    [0, 1.0, [['soma', '0.5', 'k_hh', 'gbar']]],
#    [0, 1.0, [['soma', '0.5', 'pas', 'g']]]
#]
variables = [
                [0, 1.0, [['soma', '0.5', 'ka', 'gbar']]],
                [0, 1.0, [['soma', '0.5', 'pas', 'g']]],
                [0, 1.0, [['soma', '0.5', 'ih_slow', 'gbar']]],
                [0, 1.0, [['soma', '0.5', 'nap', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'nat', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'na8st', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'narsg', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'kdr', 'gbar']]],
                [0, 1.0, [['soma', '0.5', 'ih_fast', 'gbar']]]
            ]
lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)

fitter_params = {
                    'name': 'HodgkinHuxleyFitter',
                    'variable_keys': variable_keys,
                    'errfun_name': 'rms',
                    'fitfun_names': ['get_v'],
                    'fitnessweights': [1],
                    #'model_dir': '../../../model/cells/hhCell.json',
                    #'mechanism_dir': '../../../model/channels/hodgkinhuxley',
                    #'data_dir': '../../../data/toymodels/hhCell/ramp.csv',
                    #'simulation_params': {'celsius': 6.3}
                    'model_dir': '../../../model/cells/dapmodel_simpel.json',
                    'mechanism_dir': '../../../model/channels/stellate',
                    'data_dir': '../../../data/2015_08_06d/raw/plot_IV/-0.15(nA).csv',
                    'simulation_params': {'celsius': 35}
                }

fitter = HodgkinHuxleyFitter(**fitter_params)

# save all information
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir+'/fitter.json', 'w') as f:
    json.dump(fitter_params, f, indent=4)
with open(save_dir+'/cell.json', 'w') as f:
    json.dump(Cell.from_modeldir(fitter_params['model_dir']).get_dict(), f, indent=4)

# get current traces
v_exp = fitter.data.v.values
t_exp = fitter.data.t.values
i_exp = fitter.data.i.values
dt = t_exp[1] - t_exp[0]
dvdt = np.concatenate((np.array([(v_exp[1]-v_exp[0])/dt]), np.diff(v_exp) / dt))
candidate = np.ones(len(fitter.variable_keys))
fitter.update_cell(candidate)
channel_list = get_channel_list(fitter.cell, 'soma')
ion_list = get_ionlist(channel_list)
celsius = fitter.simulation_params['celsius']

currents = currents_given_v(v_exp, t_exp, fitter.cell.soma, channel_list, ion_list, celsius)

# convert units
cell_area = get_cellarea(convert_from_unit('u', fitter.cell.soma.L),
                         convert_from_unit('u', fitter.cell.soma.diam))  # m**2
Cm = convert_from_unit('c', fitter.cell.soma.cm) * cell_area  # F
i_inj = convert_from_unit('n', fitter.data.i.values)  # A
currents = convert_from_unit('da', currents) * cell_area  # A

# linear regression
if with_cm:
    weights_adjusted, weights, residual, y, X = linear_regression(dvdt, i_inj, currents, i_pas=0, Cm=None,
                                                                  cell_area=cell_area)
else:
    weights, residual, y, X = linear_regression(dvdt, i_inj, currents, i_pas=0, Cm=Cm)

# output
if with_cm:
    print 'cm: ' + str(weights_adjusted[-1])
    print 'vclamp: ' + str(channel_list)
    print 'weights: ' + str(weights_adjusted[:-1])
else:
    print 'vclamp: ' + str(channel_list)
    print 'weights: ' + str(weights)
print 'residual: ' + str(residual)

# plot fit
plot_fit(y, X, weights, t_exp, channel_list)

# save
np.savetxt(save_dir+'/best_candidate.txt', weights)
np.savetxt(save_dir+'/error.txt', np.array([residual]))

# simulate
if with_cm:
    fitter.cell.update_attr(['soma', 'cm'], weights_adjusted[-1])
    candidate = sort_weights_by_variable_keys(channel_list, weights_adjusted[:-1], variable_keys)
    v, t, i = fitter.simulate_cell(candidate)
else:
    candidate = sort_weights_by_variable_keys(channel_list, weights, variable_keys)
    v, t, i = fitter.simulate_cell(candidate)


pl.figure()
pl.plot(t, fitter.data.v, 'k')
pl.plot(t, v, 'r')
pl.show()