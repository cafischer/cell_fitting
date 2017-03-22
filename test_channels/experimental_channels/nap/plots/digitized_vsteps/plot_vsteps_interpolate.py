import matplotlib.pyplot as pl
import numpy as np
import os
from test_channels.experimental_channels.build_trace_data import (load_traces, interpolate_traces,
                                                                  append_prepost_potential)


vsteps = np.arange(-60, -34, 5)
save_dir = './'
filenames = ['nap'+str(i)+'trace.csv' for i in vsteps]

all_traces = load_traces(save_dir, filenames, vsteps)

dt = 0.1
all_traces = interpolate_traces(all_traces, dt)

#all_traces = append_prepost_potential(all_traces, 0, 0, 30, 30, dt)


pl.figure()
for column in all_traces.columns:
    not_nan = np.logical_not(all_traces[column].isnull().values)
    pl.plot(all_traces.index[not_nan], all_traces[column][not_nan], label=column)
pl.ylabel('Current (pA)', fontsize=16)
pl.xlabel('Time (ms)', fontsize=16)
pl.legend(fontsize=16)
pl.savefig(os.path.join(save_dir+'traces_i.png'))
pl.show()

pl.figure()
for column in all_traces.columns:
    not_nan = np.logical_not(all_traces[column].isnull().values)
    min_trace = np.min(-1*all_traces[column][not_nan])
    trace = (-1 * all_traces[column][not_nan] - min_trace)
    pl.plot(all_traces.index[not_nan], trace, label=column)
pl.ylabel('Current (pA)', fontsize=16)
pl.xlabel('Time (ms)', fontsize=16)
pl.legend(fontsize=16)
pl.savefig(os.path.join(save_dir+'traces_offset_i.png'))
pl.show()

pl.figure()
for column in all_traces.columns:
    not_nan = np.logical_not(all_traces[column].isnull().values)
    min_trace = np.min(-1*all_traces[column][not_nan])
    log_trace = np.log(-1 * all_traces[column][not_nan] - min_trace)
    min_logtrace = np.min(log_trace)
    pl.plot(all_traces.index[not_nan], log_trace, label=column)
pl.ylabel('Current (pA)', fontsize=16)
pl.xlabel('Time (ms)', fontsize=16)
pl.legend(fontsize=16)
pl.savefig(os.path.join(save_dir+'traces_log_i.png'))
pl.show()

pl.figure()
for column in all_traces.columns:
    not_nan = np.logical_not(all_traces[column].isnull().values)
    min_trace = np.min(-1*all_traces[column][not_nan])
    log_trace = np.log(-1 * all_traces[column][not_nan] - min_trace)
    log_trace[log_trace<1.1] = 1 + 0.1
    min_logtrace = 1
    log_trace = (log_trace - min_logtrace)
    pl.plot(all_traces.index[not_nan], log_trace, label=column)
pl.ylabel('Current (pA)', fontsize=16)
pl.xlabel('Time (ms)', fontsize=16)
pl.legend(fontsize=16)
pl.savefig(os.path.join(save_dir+'traces_log_offset_i.png'))
pl.show()

#all_traces.to_csv(os.path.join(save_dir, 'traces.csv'))