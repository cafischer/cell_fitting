import os

import pandas as pd
from nrn_wrapper import Cell

from cell_fitting.optimization.simulate import extract_simulation_params, simulate_gates

if __name__ == '__main__':
    # parameters
    data_dir = '../../data/2015_08_26b/vrest-75/simulate_rampIV/3.0(nA).csv'
    save_dir = '../../results/server/2017-08-30_09:50:28/194/L-BFGS-B/'
    model_dir = os.path.join(save_dir, 'model', 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # get simulation_params
    data = pd.read_csv(data_dir)
    simulation_params = extract_simulation_params(data)

    # plot gates
    simulate_gates(cell, simulation_params, plot=True)