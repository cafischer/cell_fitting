import pandas as pd
import numpy as np

from cell_characteristics.analyze_APs import get_v_rest

__author__ = 'caro'

data_real_dir = '../2015_08_11d/ramp/ramp.csv'
data_new_dir = './vrest.csv'

data_real = pd.read_csv(data_real_dir)

vrest = get_v_rest(data_real.v.values, data_real.i.values)

v = np.ones(len(data_real.v.values)) * vrest
t = data_real.t.values
i = np.zeros(len(data_real.i.values))
sec = data_real.sec.values

data = pd.DataFrame({'v': v, 't': t, 'i': i, 'sec': sec})
data.to_csv(data_new_dir, index=None)



