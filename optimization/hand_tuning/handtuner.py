from PyQt5.QtWidgets import QApplication
import sys
from optimization.hand_tuning.controller import HandTuner
from optimization.helpers import *

__author__ = 'caro'

if __name__ == '__main__':

    # create app
    app = QApplication(sys.argv)

    # create problem
    variables = [
            [0, 1.0, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 1.0, [['soma', '0.5', 'nat', 'gbar']]],
            [0, 0.01, [['soma', '0.5', 'pas', 'g']]]
            ]

    lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)

    fitter_params = {
          'variable_keys': variable_keys,
          'errfun_name': 'rms',
          'fitfun_names': ['get_v'],
          'fitnessweights': [1],
          'model_dir': '../../model/cells/dapmodel_simpel.json',
          'mechanism_dir': '../../model/channels/stellate',
          'data_dir': '../../data/2015_08_26b/raw/rampIV/3.0(nA).csv',
          'simulation_params': {}
         }

    # create widget
    precision_slds = [1e-5, 1e-5, 1e-5]
    save_dir = '../../results/hand_tuning/test'
    ex = HandTuner(save_dir, fitter_params, precision_slds, lower_bounds, upper_bounds)
    sys.exit(app.exec_())