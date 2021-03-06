import os
import sys
from PyQt5.QtWidgets import QApplication
from cell_fitting.optimization.hand_tuning.controller import HandTuner
from cell_fitting.optimization.helpers import *
from cell_fitting.optimization.fitter.read_data import get_sweep_index_for_amp

__author__ = 'caro'


def get_init_var_from_model(model_dir, mechanism_dir, variables, variable_keys):
    from nrn_wrapper import Cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    init_var = np.zeros(len(variables))
    for i in range(len(variables)):
            init_var[i] = cell.get_attr(variable_keys[i][0])
    return init_var


if __name__ == '__main__':

    # create app
    app = QApplication(sys.argv)

    # fitter parameter
    variables = [
        [0.3, 2, [['soma', 'cm']]],
        # [-95, -70, [['soma', '0.5', 'pas', 'e']]],
        # [-30, -10, [['soma', '0.5', 'hcn_slow', 'ehcn']]],
        #
        # [-0.1, 0.5, [['soma', '0.5', 'pas', 'g']]],
        # [0, 0.5, [['soma', '0.5', 'nat', 'gbar']]],
        # [-0.01, 0.1, [['soma', '0.5', 'kdr', 'gbar']]],
        # [0, 1.0, [['soma', '0.5', 'nap', 'gbar']]],
        [0, 0.001, [['soma', '0.5', 'hcn_slow', 'gbar']]],
        #
        # [-100, 0, [['soma', '0.5', 'nat', 'm_vh']]],
        # [-100, 0, [['soma', '0.5', 'nat', 'h_vh']]],
        [-100, 0, [['soma', '0.5', 'kdr', 'n_vh']]],
        [-100, 0, [['soma', '0.5', 'nap', 'm_vh']]],
        [-100, 0, [['soma', '0.5', 'nap', 'h_vh']]],
        # [-100, 0, [['soma', '0.5', 'hcn_slow', 'n_vh']]],
        #
        #[1, 30, [['soma', '0.5', 'nat', 'm_vs']]],
        #[-30, -1, [['soma', '0.5', 'nat', 'h_vs']]],
        [1, 30, [['soma', '0.5', 'kdr', 'n_vs']]],
        [1, 30, [['soma', '0.5', 'nap', 'm_vs']]],
        [-30, -1, [['soma', '0.5', 'nap', 'h_vs']]],
        # [-30, -1, [['soma', '0.5', 'hcn_slow', 'n_vs']]],
        #
        # [0, 50, [['soma', '0.5', 'nat', 'm_tau_min']]],
        # [0, 50, [['soma', '0.5', 'nat', 'h_tau_min']]],
        # [0, 50, [['soma', '0.5', 'kdr', 'n_tau_min']]],
        [0, 50, [['soma', '0.5', 'nap', 'm_tau_min']]],
        [0, 50, [['soma', '0.5', 'nap', 'h_tau_min']]],
        # [0, 50, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]],

        #[0, 100, [['soma', '0.5', 'nat', 'm_tau_max']]],
        #[0, 100, [['soma', '0.5', 'nat', 'h_tau_max']]],
        # [0, 100, [['soma', '0.5', 'kdr', 'n_tau_max']]],
        [0, 100, [['soma', '0.5', 'nap', 'm_tau_max']]],
        [0, 100, [['soma', '0.5', 'nap', 'h_tau_max']]],
        # [0, 500, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]],
        #
        # [0, 10, [['soma', '0.5', 'nat', 'm_tau_delta']]],
        # [0, 10, [['soma', '0.5', 'nat', 'h_tau_delta']]],
        # [0, 10, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
        [0, 10, [['soma', '0.5', 'nap', 'm_tau_delta']]],
        [0, 10, [['soma', '0.5', 'nap', 'h_tau_delta']]],
        # [0, 10, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
    ]

    lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
    save_dir = '/home/cfischer/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    #model_dir = '../../results/server/2017-07-24_13:59:54/21/L-BFGS-B/model/cell.json'
    #model_dir = '../../results/hand_tuning/test0/cell.json'
    mechanism_dir = '../../model/channels/vavoulis'
    init_var = get_init_var_from_model(model_dir, mechanism_dir, variables, variable_keys)
    # data_read_dict = {'data_dir': '../../data/dat_files', 'cell_id': '2015_08_26b',
    #                   'protocol': 'IV', 'sweep_idx': get_sweep_index_for_amp(amp=0.4, protocol='IV'),
    #                   'v_rest_shift': -16, 'file_type': 'dat'}
    data_read_dict = {'data_dir': '../../data/dat_files', 'cell_id': '2015_08_26b',
                      'protocol': 'rampIV', 'sweep_idx': get_sweep_index_for_amp(amp=3.1, protocol='rampIV'),
                      'v_rest_shift': -16, 'file_type': 'dat'}

    fitter_params = {
        'name': 'HodgkinHuxleyFitter',
        'variable_keys': variable_keys,
        'errfun_name': 'rms',
        'fitfun_names_per_data_set': [['get_v']],
        'fitnessweights_per_data_set': [[1]],
        'model_dir': model_dir,
        'mechanism_dir': None,
        'data_read_dict_per_data_set': [data_read_dict],
        'init_simulation_params': {'celsius': 35, 'onset': 200, 'v_init': -75},
        'args': {}
    }

    # create widget
    precision_slds = [1e-5, 1e-6] + [1e-6] * (len(variables) - 2)
    save_dir = '../../results/hand_tuning/model3_0'
    ex = HandTuner(save_dir, fitter_params, precision_slds, lower_bounds, upper_bounds, init_var)
    sys.exit(app.exec_())