import os
import pandas as pd
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import extract_simulation_params, simulate_gates
from cell_fitting.util import merge_dicts


if __name__ == '__main__':
    # parameters
    data_dir = '../../data/cell_csv_data/2015_08_26b/rampIV/3.1(nA).csv'
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # get simulation_params
    data = pd.read_csv(data_dir)
    sim_params = {'onset': 200, 'v_init': -75}
    simulation_params = merge_dicts(extract_simulation_params(data.v.values, data.t.values, data.i.values), sim_params)

    # plot gates
    simulate_gates(cell, simulation_params, plot=True)