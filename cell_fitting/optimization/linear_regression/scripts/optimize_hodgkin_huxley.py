from cell_fitting.optimization.helpers import *
from cell_fitting.optimization.linear_regression import *
from cell_fitting.optimization.simulate import extract_simulation_params
from cell_fitting.optimization.linear_regression.scripts import fit_with_linear_regression
__author__ = 'caro'


if __name__ == '__main__':
    # parameter
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/linear_regression/hodgkin_huxley/'
    model_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/cells/hhCell.json'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/hodgkinhuxley'
    with_cm = True
    init_simulation_params = {'celsius': 6.3}

    variables = [
       [0, 1.0, [['soma', '0.5', 'na_hh', 'gbar']]],
       [0, 1.0, [['soma', '0.5', 'k_hh', 'gbar']]],
       [0, 1.0, [['soma', '0.5', 'pas', 'g']]]
    ]
    _, _, variable_keys = get_lowerbound_upperbound_keys(variables)

    # load data
    data_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/toymodels/hhCell/rampIV/amp_3.0'
    v_exp = np.load(os.path.join(data_dir, 'v.npy'))
    t_exp = np.load(os.path.join(data_dir, 't.npy'))
    i_exp = np.load(os.path.join(data_dir, 'i_inj.npy'))

    simulation_params = extract_simulation_params(v_exp, t_exp, i_exp, **init_simulation_params)

    fit_with_linear_regression(v_exp, t_exp, i_exp, save_dir, model_dir, mechanism_dir, variable_keys,
                               simulation_params, with_cm)