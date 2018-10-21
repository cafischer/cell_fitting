import matplotlib.pyplot as pl
import numpy as np
import os
from test_channels.experimental_channels.build_trace_data import (load_traces, interpolate_traces,
                                                                  append_prepost_potential)


vsteps = np.arange(-60, -34, 5)
save_dir = './'
filenames = ['nap'+str(i)+'trace.csv' for i in vsteps]

all_traces = load_traces(save_dir, filenames, vsteps)

#dt = 0.1
#all_traces = interpolate_traces(all_traces, dt, 1)


pl.figure()
for column in all_traces.columns:
    not_nan = np.logical_not(all_traces[column].isnull().values)
    pl.plot(all_traces.index[not_nan], all_traces[column][not_nan], label=column)
pl.ylabel('Current (pA)', fontsize=16)
pl.xlabel('Time (ms)', fontsize=16)
pl.legend(fontsize=16)
pl.savefig(os.path.join(save_dir+'traces.png'))
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
pl.savefig(os.path.join(save_dir+'traces_offset.png'))
pl.show()

pl.figure()
for column in all_traces.columns:
    not_nan = np.logical_not(all_traces[column].isnull().values)
    min_trace = np.min(-1*all_traces[column][not_nan])
    log_trace = np.log(-1 * all_traces[column][not_nan] - min_trace)
    min_logtrace = np.min(log_trace)
    pl.plot(all_traces.index[not_nan], log_trace, label=column)
pl.ylabel('Log Current (pA)', fontsize=16)
pl.xlabel('Time (ms)', fontsize=16)
pl.legend(fontsize=16)
pl.savefig(os.path.join(save_dir+'traces_log.png'))
pl.show()

pl.figure()
from scipy.stats import linregress
cmap = pl.get_cmap('Vega10')
colors = cmap(np.linspace(0, 1, len(all_traces.columns)))
for i, column in enumerate(all_traces.columns):
    not_nan = np.logical_not(all_traces[column].isnull().values)
    min_trace = np.min(-1*all_traces[column][not_nan])
    log_trace = np.log(-1 * all_traces[column][not_nan] - min_trace)
    not_inf = np.logical_not(log_trace.values == -np.inf)
    t = all_traces.index[not_nan].values
    slope, intercept, _, _, _ = linregress(t[not_inf], log_trace.values[not_inf])
    print 'slope: ', -1.0/slope
    pl.plot(t, log_trace, label=int(np.round(-1.0/slope, 0)), color=colors[i])
    pl.plot(t, t * slope + intercept, color=colors[i])
pl.ylabel('Log Current (pA)', fontsize=16)
pl.xlabel('Time (ms)', fontsize=16)
pl.ylim([-0.5, 7.5])
legend = pl.legend(fontsize=16, title='Slopes')
pl.setp(legend.get_title(),fontsize=16)
pl.savefig(os.path.join(save_dir+'linear_fit_to_log_traces.png'))
pl.show()


all_traces.to_csv(os.path.join(save_dir, 'traces.csv'))