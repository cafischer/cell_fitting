import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as pl
from data import change_dt

__author__ = 'caro'

data_dirs = [
            './2015_08_11d/step/stepcurrent-0.1.csv',
            './2015_08_11d/ramp/ramp.csv',
            './2015_08_11d/zap/zap.csv'
             ]
data_new_dir = './2015_08_11d/merged/'

data_sets = list()
dts = list()
for i, data_dir in enumerate(data_dirs):
    data_sets.append(pd.read_csv(data_dir))
    dts.append(data_sets[i].t[1])

dt = np.max(dts)
for i in range(len(data_sets)):
    data_sets[i] = change_dt(dt, data_sets[i])
    if i > 0:
        data_sets[i].t += data_sets[i-1].t.values[-1] + dt

    print data_sets[i].t.values[-1] / dt


if not os.path.exists(data_new_dir):
    os.makedirs(data_new_dir)

data_merged = pd.concat(data_sets, ignore_index=True)

pl.figure()
pl.plot(data_merged.t, data_merged.v)
pl.show()
pl.figure()
pl.plot(data_merged.t, data_merged.i)
pl.show()

data_merged.to_csv(data_new_dir+'step_dap_zap.csv', index=False)



