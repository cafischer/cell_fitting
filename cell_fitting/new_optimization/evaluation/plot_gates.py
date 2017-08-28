import pandas as pd
import os
from cell_fitting.optimization.simulate import extract_simulation_params, simulate_gates
from nrn_wrapper import Cell


if __name__ == '__main__':
    # parameters
    data_dir = '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv'
    save_dir = '../../results/server/2017-08-23_08:41:41/270/L-BFGS-B/'
    model_dir = os.path.join(save_dir, 'model', 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # get simulation_params
    data = pd.read_csv(data_dir)
    simulation_params = extract_simulation_params(data)

    # plot gates
    simulate_gates(cell, simulation_params, plot=True)