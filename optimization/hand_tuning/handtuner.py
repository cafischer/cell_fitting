from PyQt5.QtWidgets import QApplication
import sys

from optimization.hand_tuning.controller import HandTuner

__author__ = 'caro'

if __name__ == '__main__':

    # create app
    app = QApplication(sys.argv)

    # create problem
    variables = [
            [0, 2.5, [['soma', '0.5', 'na8st', 'gbar']]],
            [0, 2.5, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 2.5, [['soma', '0.5', 'pas', 'g']]]
            ]

    problem_dict = {
          'name': 'CellFitProblem',
          'maximize': False,
          'normalize': True,
          'model_dir': '../../model/cells/toymodel3.json',
          'mechanism_dir': '../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../data/2015_08_11d/ramp/dap.csv',  #'../../data/toymodels/toymodel3/ramp.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

    # create widget
    precision_slds = [1e-5, 1e-5, 1e-5]
    save_dir = '../../results/hand_tuning/test'
    ex = HandTuner(save_dir, problem_dict, precision_slds)
    sys.exit(app.exec_())

    # TODO: add xlim, ylim