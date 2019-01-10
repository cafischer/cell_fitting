from cell_fitting.optimization.helpers import *
from cell_fitting.optimization.simulate import  extract_simulation_params
from cell_fitting.optimization.fitter.read_data import get_sweep_index_for_amp, read_data
from cell_fitting.optimization.linear_regression.scripts import fit_with_linear_regression

__author__ = 'caro'

# parameter
save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/linear_regression/fit_with_stellate_channels_modeldb/'
model_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/cells/dapmodel_simpel.json'
mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/stellate_channels_modeldb'
with_cm = True
init_simulation_params = {'celsius': 35}

# variables = [
#                 [0, 1.0, [['soma', '0.5', 'ka', 'gbar']]],
#                 [0, 1.0, [['soma', '0.5', 'pas', 'g']]],
#                 [0, 1.0, [['soma', '0.5', 'ih_fast', 'gbar']]],
#                 [0, 1.0, [['soma', '0.5', 'ih_slow', 'gbar']]],
#                 [0, 1.0, [['soma', '0.5', 'nap', 'gbar']]],
#                 [0, 1.0, [['soma', '0.5', 'nat', 'gbar']]],
#                 [0, 1.0, [['soma', '0.5', 'kdr', 'gbar']]],
#             ]

variables = [
                [0, 1.0, [['soma', '0.5', 'pas', 'g']]],

                #[0, 1.0, [['soma', '0.5', 'hcn_fast_sh13', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'hcn_slow_sh13', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'kap_sh13', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'kdr_sh13', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'nap_sh13', 'gbar']]],
                [0, 1.0, [['soma', '0.5', 'nax_sh13', 'gbar']]],

                [0, 1.0, [['soma', '0.5', 'h_j17', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'kap_j17', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'kdr_j17', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'nax_j17', 'gbar']]],

                #[0, 1.0, [['soma', '0.5', 'hcn_fast_sh17', 'gbar']]],
                #[0, 1.0, [['soma', '0.5', 'hcn_slow_sh17', 'gbar']]],
                [0, 1.0, [['soma', '0.5', 'km_sh17', 'gbar']]],
                [0, 1.0, [['soma', '0.5', 'kap_sh17', 'gbar']]],
                [0, 1.0, [['soma', '0.5', 'kdr_sh17', 'gbar']]],
                [0, 1.0, [['soma', '0.5', 'na8st_sh17', 'gbar']]],
                [0, 1.0, [['soma', '0.5', 'nap_sh17', 'gbar']]],
            ]

lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)

# load data
protocol = 'rampIV'
data_read_dict = {'data_dir': '/home/cf/Phd/DAP-Project/cell_data/raw_data', 'cell_id': '2015_08_26b',
                  'protocol': protocol, 'sweep_idx': get_sweep_index_for_amp(amp=3.1, protocol=protocol),
                  'v_rest_shift': -16, 'file_type': 'dat'}
data = read_data(**data_read_dict)
v_exp = data['v']
t_exp = data['t']
i_exp = data['i_inj']
simulation_params = extract_simulation_params(v_exp, t_exp, i_exp, **init_simulation_params)

fit_with_linear_regression(v_exp, t_exp, i_exp, save_dir, model_dir, mechanism_dir, variable_keys,
                           simulation_params, with_cm)