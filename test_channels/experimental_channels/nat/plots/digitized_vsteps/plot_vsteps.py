import matplotlib.pyplot as pl
import numpy as np
import os
from test_channels.experimental_channels.build_trace_data import (load_traces, interpolate_traces,
                                                                  append_prepost_potential)


vsteps = np.arange(-30, -9, 5)
save_dir = './'
filenames = ['nat'+str(i)+'trace.csv' for i in vsteps]

all_traces = load_traces(save_dir, filenames, vsteps)

dt = 0.01
all_traces.iloc[0].fillna(0, inplace=True)
all_traces = interpolate_traces(all_traces, dt, 1)

pl.figure()
for column in all_traces.columns:
    not_nan = np.logical_not(all_traces[column].isnull().values)
    pl.plot(all_traces.index[not_nan], all_traces[column][not_nan], label=column)
pl.ylabel('Current (pA)', fontsize=16)
pl.xlabel('Time (ms)', fontsize=16)
pl.legend(fontsize=16)
pl.savefig(os.path.join(save_dir, 'traces.png'))
pl.show()

#all_traces = all_traces[all_traces.columns[-1]]

all_traces.to_csv(os.path.join(save_dir, 'traces.csv'))