from time import time
import json
import os
from functools import partial
from optimization.problems import get_lowerbound_upperbound_keys
from optimization.problems.abstract_problems import *
from optimization.problems.fit_izhikevich_model import *
from optimization.bio_inspired.optimize import optimize_bio_inspired
from optimization.errfuns import rms
from fitfuns import *

pop_size = 200
max_iterations = 50
cell = '2015_08_26b'

variables = [
    #[150, 300, 'cm'],
    #[0.1, 2, 'k_rest'],
    #[0.1, 5, 'k_t'],
    [0.0001, 0.01, 'a1'],
    [20, 40, 'b1'],
    [-5, 5, 'd1'],
    [-1, 0, 'a2'],
    [0, 20, 'b2'],
    [0, 400, 'd2']
    ]
# TODO version 2
#a1 = 0.007  # 0.017
#b1 = 30  # 27
#d1 = 0
lower_bound, upper_bound, variable_keys = get_lowerbound_upperbound_keys(variables)

v_rest = -62.5
v_peak = 51.5
given_variables = {'cm': 185, 'k_rest': 0.75, 'k_t': 200, 'a1': 0.0072, 'b1': 28.21, 'd1': 0.0,
                   'v_rest': v_rest, 'v_t': -47.0, 'v_reset': -49.0, 'v_peak': v_peak, 'i_b': 0,
                   'v0': v_rest, 'u0': [0, 0]}

data_dir = '../data/2015_08_26b/simulate_rampIV/3.0(nA).csv'
data = pd.read_csv(data_dir)
dt = data.t[1]
data_to_fit = data.v[int(13.6/dt):int(120/dt)]

fitter_specification = {
    'name': 'IzhikevichFitter',
    'variable_keys': variable_keys,
    'given_variables': given_variables,
    'data_to_fit': [data_to_fit],
    'fitfuns': [partial(get_v_DAP, v_peak=v_peak, data_to_fit=data_to_fit)],  #[get_v, get_v],
    'errfun': rms,
    'data_dirs': [data_dir]
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

save_dir = './results/PSO/'+cell+'/simulate_rampIV/DAP_version2(1)/'
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