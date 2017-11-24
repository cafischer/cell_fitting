from time import time
import json
import os
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
from cell_fitting.optimization.problems.abstract_problems import *
from cell_fitting.optimization.problems.fit_izhikevich_model import *
from cell_fitting.optimization.errfuns import rms
from fitfuns import *

pop_size = 200
max_iterations = 50
cell = '2015_08_26b'

variables = [
    [100, 350, 'cm'],
    [0.1, 2, 'k_rest'],
    [0, 0.1, 'a1'],
    [1, 40, 'b1']
    #[0, 1, 'a2'],
    #[0, 30, 'b2']
    ]
lower_bound, upper_bound, variable_keys = get_lowerbound_upperbound_keys(variables)

v_rest = -62.5
given_variables = {'v_rest': v_rest, 'v_t': -47.0, 'v_reset': -49.0, 'v_peak': 51.5, 'i_b': 0,
                   'v0': v_rest, 'u0': [0, 0], 'a2': 0, 'b2':0, 'd1': 0, 'd2': 0, 'k_t': 1}

fitter_specification = {
    'name': 'IzhikevichFitter',
    'variable_keys': variable_keys,
    'given_variables': given_variables,
    'fitfuns': [get_v, get_v],
    'errfun': rms,
    'data_dirs': ['../data/'+cell+'/IV/-0.15(nA).csv', '../data/'+cell+'/simulate_rampIV/1.0(nA).csv']
}
izhikevich_fitter = IzhikevichFitter(**fitter_specification)

problem_specification = {
    'name': 'NormalizedProblem',
    'maximize': False,
    'lower_bound': lower_bound,
    'upper_bound': upper_bound,
    'n_variables': len(variables),
    'evaluate': izhikevich_fitter.evaluate
}
problem = NormalizedProblem(**problem_specification)

#-----------------------------------------------------------------------------------

save_dir = './results/PSO/'+cell+'/IV_rampIV/-0.15(nA)_1.0(nA)/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

method = 'PSO'
method_type = 'swarm'
method_args = {'inertia': 0.43, 'cognitive_rate': 1.44, 'social_rate': 1.57}
method_args['pop_size'] = pop_size
method_args['max_generations'] = max_iterations

individuals_file = open(save_dir + '/individuals_file.csv', 'w')
seed = time()

# saving
with open(save_dir+'seed.txt', 'w') as f:
    f.write(str(seed))
#with open(save_dir + 'problem.json', 'w') as f:  TODO
#    json.dump(problem_specification, f)
#with open(save_dir + 'fitter.json', 'w') as f:
#    json.dump(fitter_specification, f)
with open(save_dir + 'method_args.json', 'w') as f:
    json.dump(method_args, f)

optimize_bio_inspired(method, method_type, method_args, problem, individuals_file, seed)