import matplotlib.pyplot as pl
import numpy as np
import os
from test_channels.experimental_channels.build_trace_data import (load_traces, interpolate_traces,
                                                                  append_prepost_potential)


vsteps = np.array([-110, -100, -90, -80, -70, -65])
save_dir = './'
filenames = ['HCN'+str(i)+'trace.csv' for i in vsteps]

all_traces = load_traces(save_dir, filenames, vsteps)
all_traces *= -1
all_traces -= 1187

dt = 0.01
all_traces.iloc[0].fillna(0, inplace=True)
all_traces = interpolate_traces(all_traces, dt, 1)

all_traces = all_traces[all_traces.index >= 8]

#all_traces = append_prepost_potential(all_traces, 0, 0, 30, 30, dt)

pl.figure()
for column in all_traces.columns:
    not_nan = np.logical_not(all_traces[column].isnull().values)
    pl.plot(all_traces.index[not_nan], all_traces[column][not_nan], label=column)
pl.ylabel('Current (pA)', fontsize=16)
pl.xlabel('Time (ms)', fontsize=16)
pl.legend(fontsize=16)
pl.savefig(os.path.join(save_dir, 'traces.png'))
pl.show()


all_traces.to_csv(os.path.join(save_dir, 'traces.csv'))