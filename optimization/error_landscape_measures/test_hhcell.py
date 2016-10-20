from time import time
import functools
import json
import os
from random import Random
import pandas as pd
from optimization.error_landscape_measures import *
from optimization.bio_inspired.inspyred_extension.generators import *
from optimization.helpers import get_lowerbound_upperbound_keys
from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter
from optimization.fitfuns import *


def remap(dictionary):
    return [{'key': key, 'value': value} for key, value in dictionary.iteritems()]

start_time = time()

# parameter
optimum = np.array([0.12, 0.036, 0.0003])
#save_dir = '../../results/fitness_landscape_analysis/hhCell/3params/rms/'
save_dir = '../../results/fitness_landscape_analysis/test/'
variables = [
            [0, 1.5, [['soma', '0.5', 'na_hh', 'gnabar']]],
            [0, 1.5, [['soma', '0.5', 'k_hh', 'gkbar']]],
            [0, 1.5, [['soma', '0.5', 'pas', 'g']]]
            ]
model_dir = '../../model/cells/hhCell.json'
mechanism_dir = '../../model/channels/hodgkinhuxley'
data_dir = '../../data/toymodels/hhCell/ramp.csv'
n_candidates = 100
n_repeat = 10
n_redrawn_candidates = 50
radius = 10 ** (-2)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# generate candidates
lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)

# fitness function
errfun = 'rms'
fitfun = 'shifted_AP'
#fitfun = 'get_v'
fitnessweights = [1]
simulation_params = {'celsius': 6.3}
data = pd.read_csv(data_dir)
APtime = get_APtime(np.array(data.v), np.array(data.t), np.array(data.i), None)
args = {'APtime': APtime[0], 'shift': 4, 'window_size': 10}
#args = None
fitter = HodgkinHuxleyFitter(variable_keys, errfun, fitfun, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params=None, args=args)
fitness = functools.partial(fitter.evaluate_fitness, args=None)

# save
with open(save_dir+'/variables.json', 'w') as f:
    json.dump(variables, f, indent=4)
with open(save_dir+'/n_candidates.txt', 'w') as f:
    f.write(str(n_candidates))
with open(save_dir + '/n_repeat.txt', 'w') as f:
    f.write(str(n_repeat))
with open(save_dir + '/radius.txt', 'w') as f:
    f.write(str(radius))
with open(save_dir + '/optimum.npy', 'w') as f:
    np.save(f, optimum)
with open(save_dir+'/fitter.json', 'w') as f:
    json.dump(fitter.to_dict(), f, indent=4)

# assign candidates to minima
for i in range(n_repeat):
    seed = time()
    random = Random()
    random.seed(seed)
    candidates = [get_random_numbers_in_bounds(random, lower_bounds, upper_bounds, None) for j in range(n_candidates)]
    local_minimum_per_candidate = find_local_minimum_for_each_candidate(candidates, fitness,
                                                                        n_redrawn_candidates=n_redrawn_candidates,
                                                                        radius=radius)
    with open(save_dir+'local_minimum_per_candidate('+str(i)+').json', 'w') as f:
        json.dump(remap(local_minimum_per_candidate), f, indent=4)
    with open(save_dir + '/seed('+str(i)+').txt', 'w') as f:
        f.write(str(seed))


end_time = time()

print 'Time: ' + str(end_time - start_time)
