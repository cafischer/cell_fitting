from __future__ import division
import numdifftools as nd
import os
from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter
from optimization.helpers import *
import functools
import json

# parameter
save_dir = '../../results/fitnesslandscapes/gradientlandscape/gna_gk_highresolution/'
fitfun_names = ['get_APamp']
fitnessweights = [1]
p1_range = np.arange(0, 0.4, 0.0001)  # 0.12
p2_range = np.arange(0, 0.5, 0.0001)  # 0.036
chunk_size = 100

variable_keys = [[['soma', '0.5', 'na_hh', 'gnabar']],
                 [['soma', '0.5', 'k_hh', 'gkbar']]]
model_dir = '../../model/cells/hhCell.json'
mechanism_dir = '../../model/channels/hodgkinhuxley'
data_dir = '../../data/toymodels/hhCell/ramp.csv'
args = {'threshhold': -30}
if not os.path.exists(save_dir+ fitfun_names[0]):
    os.makedirs(save_dir+ fitfun_names[0])

fitter = HodgkinHuxleyFitter(variable_keys, 'rms', fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params=None, args=None)
evaluate_fitness = functools.partial(fitter.evaluate_fitness, args=args)

def jac(candidate):
    jac = nd.Jacobian(evaluate_fitness, step=1e-8, method='central')(candidate)[0]
    jac[np.isnan(jac)] = 0
    return jac

# compute error
n_chunks_p1 = len(p1_range) / chunk_size
n_chunks_p2 = len(p2_range) / chunk_size
assert n_chunks_p1.is_integer()
assert n_chunks_p2.is_integer()
n_chunks_p1 = int(n_chunks_p1)
n_chunks_p2 = int(n_chunks_p2)
for c1 in range(n_chunks_p1):
    for c2 in range(n_chunks_p2):
        p1_chunk = p1_range[c1 * chunk_size:(c1 + 1) * chunk_size]
        p2_chunk = p2_range[c2 * chunk_size:(c2 + 1) * chunk_size]
        gradientlandscape = np.zeros((len(p1_chunk), len(p2_chunk), len(variable_keys)))
        for i, p1 in enumerate(p1_chunk):
            for j, p2 in enumerate(p2_chunk):
                gradientlandscape[i, j, :] = jac([p1, p2])

        with open(save_dir + fitfun_names[0] + '/gradientlandscape'+str(c1)+'_'+str(c2)+ '.npy', 'w') as f:
            np.save(f, gradientlandscape)


# save models
with open(save_dir + fitfun_names[0] + '/chunk_size.txt', 'w') as f:
    f.write(str(chunk_size))
np.savetxt(save_dir + fitfun_names[0] + '/p1_range.txt', p1_range)
np.savetxt(save_dir + fitfun_names[0] + '/p2_range.txt', p2_range)
with open(save_dir + fitfun_names[0] + '/fitter.json', 'w') as f:
    json.dump(fitter.to_dict(), f)