import numpy as np
import matplotlib.pyplot as pl
import json
import pandas as pd

save_dir = '../../results/fitnesslandscapes/modellandscape/gna_gk'

with open(save_dir+'/modellandscape.npy', 'r') as f:
    modellandscape = np.load(f)
p1_range = np.loadtxt(save_dir+'/p1_range.txt')
p2_range = np.loadtxt(save_dir+'/p2_range.txt')
with open(save_dir+'/dirs.json', 'r') as f:
    dirs = json.load(f)

data = pd.read_csv(dirs['data_dir'])

optimum = [0.05, 0.02]  #[0.12, 0.04]  # [0.12, 0.036]
idx_optimum = [np.where(np.isclose(p1_range, optimum[0]))[0][0], np.where(np.isclose(p2_range, optimum[1]))[0][0]]

n = 5
f, ax = pl.subplots(n, n, sharex=True, sharey=True)
for ax_i, i in enumerate(range(idx_optimum[0]-(n-1)/2, idx_optimum[0]+(n-1)/2+1)):
    for ax_j, j in enumerate(range(idx_optimum[1]-(n-1)/2, idx_optimum[1]+(n-1)/2+1)):
        if i >= 0 and j>=0 and i < np.shape(modellandscape)[0] and j < np.shape(modellandscape)[1]:
            ax[n-1-ax_j, ax_i].plot(data.t, modellandscape[i, j])
#pl.savefig(save_dir + 'model_landscape.svg')
pl.show()