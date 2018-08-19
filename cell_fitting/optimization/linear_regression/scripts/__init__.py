import json
import os
import numpy as np
from nrn_wrapper import Cell
from cell_fitting.optimization.helpers import *
from cell_fitting.optimization.linear_regression import *
from cell_fitting.optimization.simulate import currents_given_v, iclamp_handling_onset
from cell_fitting.util import convert_from_unit


def get_derivative(x, dt):
    dxdt = np.concatenate((np.array([(x[1] - x[0]) / dt]), np.diff(x) / dt))
    return dxdt


def get_current_traces(cell, variable_keys, v_exp, t_exp, simulation_params):
    update_variables(cell, np.ones(len(variable_keys)), variable_keys)
    channel_list = get_channel_list(cell, 'soma')
    ion_list = get_ionlist(channel_list)
    currents = currents_given_v(v_exp, t_exp, cell.soma, channel_list, ion_list, simulation_params['celsius'])
    return currents, channel_list


def convert_units(cell, i_exp, currents):
    cell_area = get_cellarea(convert_from_unit('u', cell.soma.L),
                             convert_from_unit('u', cell.soma.diam))  # m**2
    Cm = convert_from_unit('c', cell.soma.cm) * cell_area  # F
    i_inj = convert_from_unit('n', i_exp)  # A
    currents = convert_from_unit('da', currents) * cell_area  # A
    return cell_area, Cm, i_inj, currents


def fit_with_linear_regression(v_exp, t_exp, i_exp, save_dir, model_dir, mechanism_dir, variable_keys,
                               simulation_params, with_cm):

    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    cell.insert_mechanisms(variable_keys)

    dvdt = get_derivative(v_exp, t_exp[1] - t_exp[0])
    currents, channel_list = get_current_traces(cell, variable_keys, v_exp, t_exp, simulation_params)
    cell_area, Cm, i_inj, currents = convert_units(cell, i_exp, currents)
    if with_cm:
        Cm = None
    weights_adjusted, weights, residual, y, X = linear_regression(dvdt, i_inj, currents, i_pas=0, Cm=Cm,
                                                                  cell_area=cell_area)

    # simulate
    if with_cm:
        cell.update_attr(['soma', 'cm'], weights_adjusted[-1])
        candidate = sort_weights_by_variable_keys(channel_list, weights_adjusted[:-1], variable_keys)
        update_variables(cell, candidate, variable_keys)
        v, t, i = iclamp_handling_onset(cell, **simulation_params)
    else:
        candidate = sort_weights_by_variable_keys(channel_list, weights, variable_keys)
        update_variables(cell, candidate, variable_keys)
        v, t, i = iclamp_handling_onset(cell, **simulation_params)

    # output
    if with_cm:
        print 'cm: ' + str(weights_adjusted[-1])
        print 'vclamp: ' + str(channel_list)
        print 'weights: ' + str(weights_adjusted[:-1])
    else:
        print 'vclamp: ' + str(channel_list)
        print 'weights: ' + str(weights)
    print 'residual: ' + str(residual)

    # save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'cell.json'), 'w') as f:
        json.dump(cell.get_dict(), f, indent=4)
    with open(os.path.join(save_dir, 'variables_keys.json'), 'w') as f:
        json.dump(variable_keys, f, indent=4)
    np.savetxt(os.path.join(save_dir, 'best_candidate.txt'), weights)
    np.savetxt(os.path.join(save_dir, 'error.txt'), np.array([residual]))

    # plot fit
    plot_fit(y, X, weights, t_exp, channel_list, save_dir=save_dir)

    pl.figure()
    pl.plot(t_exp, v_exp, 'k')
    pl.plot(t, v, 'r')
    pl.xlabel('Mem. pot. (mV)')
    pl.ylabel('Time (ms)')
    pl.savefig(os.path.join(save_dir, 'simulation.png'))
    pl.show()
