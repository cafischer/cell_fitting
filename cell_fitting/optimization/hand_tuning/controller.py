import json
import os
import numpy as np

from cell_fitting.optimization.hand_tuning.model import Model
from cell_fitting.optimization.hand_tuning.view import ViewHandTuner

__author__ = 'caro'


class HandTuner:

    def __init__(self, save_dir, fitter_params, precision_slds, lower_bounds, upper_bounds, init_vals=None):

        self.save_dir = save_dir

        # model
        self.model = Model(fitter_params)

        # view
        name_variables = [p[0][-2] + ' ' + p[0][-1] for p in self.model.fitter.variable_keys]
        lower_bounds = lower_bounds
        upper_bounds = upper_bounds
        slider_fun = self.update_img
        button_names = ['Reset Traces', 'Save Cell']
        button_funs = [self.reset_all_imgs, self.save_cell]
        self.view = ViewHandTuner(name_variables, precision_slds, lower_bounds, upper_bounds, slider_fun,
                                  button_names, button_funs, init_vals)
        self.reset_img(0)
        self.reset_img(1)
        for i in range(len(name_variables)):
            self.model.update_var(i, init_vals[i])

    def save_cell(self):
        # create folders
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # save cell dictionary
        with open(self.save_dir+'/cell.json', 'w') as f:
            json.dump(self.model.fitter.cell.get_dict(), f, indent=4)

    def reset_img(self, img_id):
        self.view.clear_ax(img_id)
        t = np.arange(0, self.model.fitter.simulation_params[0]['tstop']+self.model.fitter.simulation_params[0]['dt'],
                      self.model.fitter.simulation_params[0]['dt'])
        if img_id == 0:
            self.view.plot_on_ax(img_id, t, self.model.fitter.data_sets_to_fit[0][0], color='k',
                                 xlabel='Time (ms)', ylabel='V (mV)', linewidth=1.5)
        elif img_id == 1:
            self.view.plot_on_ax(img_id, t, self.model.lhsHH, color='k',
                                 xlabel='Time (ms)', ylabel='$c_m \cdot dV/dt - i_{inj}$', linewidth=1.5)
        self.view.plot_img(img_id)

    def reset_all_imgs(self):
        for i in range(len(self.view.imgs)):
            self.reset_img(i)

    def update_img(self, value):

        if value is not None:
            # identify sender (parameter to update)
            sender = self.view.sender()
            id = sender.id

            # transform value to the range of the specific parameter
            value_unnorm = value * self.view.slds[id].precision
            sender.value.setText('Value: '+str(value_unnorm))

            # update parameter that caused the event
            self.model.update_var(id, value_unnorm)

        # run simulations
        v, t = self.model.simulate()
        currents = self.model.get_current()
        rhsHH = self.model.get_rhsHH(currents)

        # update plots
        self.view.plot_on_ax(0, t, v, xlabel='Time (ms)', ylabel='V (mV)', linewidth=1.5)  # no reset to see how successive changes affect the trace
        self.view.plot_img(0)

        self.reset_img(1)  # needs to be reset for clarity (different currents have different colors)
        self.view.plot_on_ax(1, t, rhsHH, color='0.5', label='$-\sum_{ion} i_{ion}$',
                             xlabel='Time (ms)', ylabel='$c_m \cdot dV/dt - i_{inj}$', linewidth=1.5)
        for i, current in enumerate(currents):
            self.view.plot_on_ax(1, t, -current, label=self.model.channel_list[i],
                                 xlabel='Time (ms)', ylabel='$c_m \cdot dV/dt - i_{inj}$')
        self.view.plot_img(1)