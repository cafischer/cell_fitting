from PyQt5.QtWidgets import QApplication
import sys
from optimization.hand_tuning.controller import HandTuner
from optimization.helpers import *

__author__ = 'caro'

if __name__ == '__main__':

    # create app
    app = QApplication(sys.argv)

    # fitter parameter
    variables = [
            #[0, 0.1, [['soma', '0.5', 'nap_act', 'gbar']]],
            #[-100, 0, [['soma', '0.5', 'nap_act', 'm_vh']]],
            #[1, 30, [['soma', '0.5', 'nap_act', 'm_vs']]],
            [0, 0.1, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 0.01, [['soma', '0.5', 'pas', 'g']]],
            [0.5, 1.5, [['soma', 'cm']]]
            ]
    init_var = [0.0446640321206, 0.00474507123453, 0.7]
    lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)

    fitter_params = {
        'name': 'HodgkinHuxleyFitter',
        'variable_keys': variable_keys,
        'errfun_name': 'rms',
        'fitfun_names': ['get_v'],
        'fitnessweights': [1],
        'model_dir': '../../model/cells/2017-06-19_13:12:49_189.json',
        'mechanism_dir': '../../model/channels/vavoulis',
        'data_dir': '../../data/2015_08_26b/vrest-75/IV/0.4(nA).csv',
        'simulation_params': {'celsius': 35, 'onset': 200},
        'args': {}
    }

    # create widget
    precision_slds = [1e-5, 1e-5, 1e-3]
    save_dir = '../../results/hand_tuning/test0'
    ex = HandTuner(save_dir, fitter_params, precision_slds, lower_bounds, upper_bounds, init_var)
    sys.exit(app.exec_())