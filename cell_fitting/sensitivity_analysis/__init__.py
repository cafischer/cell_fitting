import os

import numpy as np
import pandas as pd
from nrn_wrapper import Cell

from cell_fitting.optimization import create_pseudo_random_number_generator
from cell_fitting.optimization.bio_inspired import get_random_numbers_in_bounds
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
from cell_fitting.optimization.simulate import iclamp_handling_onset, extract_simulation_params
from cell_fitting.util import merge_dicts


def get_cell(model_dir, mechanism_dir, variable_keys):
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    cell.insert_mechanisms(variable_keys)
    return cell


def update_cell(cell, candidate, variable_keys):
    for i in range(len(candidate)):
        for path in variable_keys[i]:
            cell.update_attr(path, candidate[i])


def simulate_random_candidates(save_dir, n_candidates, seed, model_dir, mechanism_dir, variables, data_dir,
                               init_simulation_params):

    # get lower upper bounds
    lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)

    # create pseudo_random_number_generator
    prng = create_pseudo_random_number_generator(seed)

    # create cell
    cell = get_cell(model_dir, mechanism_dir, variable_keys)

    # get simulation params
    data = pd.read_csv(data_dir)
    simulation_params = merge_dicts(extract_simulation_params(data), init_simulation_params)

    for i_candidate in range(n_candidates):

        # generate new candidate
        candidate = get_random_numbers_in_bounds(prng, lower_bounds, upper_bounds, None)
        update_cell(cell, candidate, variable_keys)

        # simulate candidate
        v, t, i_inj = iclamp_handling_onset(cell, **simulation_params)

        # save
        candidate_dir = os.path.join(save_dir, str(i_candidate))
        if not os.path.exists(candidate_dir):
            os.makedirs(candidate_dir)

        with open(os.path.join(candidate_dir, 'candidate.npy'), 'w') as f:
            np.save(f, candidate)
        with open(os.path.join(candidate_dir, 'v.npy'), 'w') as f:
            np.save(f, v)

    with open(os.path.join(save_dir, 't.npy'), 'w') as f:
        np.save(f, t)

    with open(os.path.join(save_dir, 'i_inj.npy'), 'w') as f:
        np.save(f, i_inj)