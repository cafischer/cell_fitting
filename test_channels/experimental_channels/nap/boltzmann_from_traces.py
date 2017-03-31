from __future__ import division
import pandas as pd
import matplotlib.pyplot as pl
import os
import numpy as np
from test_channels.channel_characteristics import boltzmann_fun


ena = 63
traces_exp = pd.read_csv(os.path.join('.', 'plots', 'digitized_vsteps', 'traces.csv'), index_col=0)
v_range = np.array([float(c) for c in traces_exp.columns])

# steady-state: Magistretti
vh_act_exp = -44.4
k_act_exp = -5.2
vh_inact_exp = -48.8
k_inact_exp = 10
steadystate_act_exp = boltzmann_fun(v_range, vh_act_exp, k_act_exp)

# steady state from current traces
i_steadystate = np.zeros(len(traces_exp.columns))
for i, column in enumerate(traces_exp.columns):
    traces_exp[column] /= (v_range[i] - ena)
    i_steadystate[i] = traces_exp[column].values[-1]

i_steadystate /= np.max(steadystate_act_exp)

pl.figure()
pl.plot(v_range, i_steadystate, 'k', label='traces')
pl.plot(v_range, steadystate_act_exp, 'b', label='tail')
pl.ylabel('Current (normalized)', fontsize=16)
pl.xlabel('Time (ms)', fontsize=16)
pl.legend(fontsize=16, loc='lower right')
pl.show()