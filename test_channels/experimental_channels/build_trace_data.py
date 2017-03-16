import pandas as pd
import numpy as np
import os
from scipy.interpolate import CubicSpline


def load_traces(save_dir, filenames, vsteps, lower_zero=True):
    traces = pd.DataFrame(columns=['t'])
    for filename in filenames:
        trace = pd.read_csv(os.path.join(save_dir, filename), header=None, names=['t', 'i'])
        trace['i'] *= -1  # upside down
        traces = traces.merge(trace, on='t', how='outer')

    if lower_zero:
        traces -= np.nanmax(traces.values)  # adjust zero level
    else:
        traces -= np.nanmin(traces.values)  # adjust zero level
    traces.sort_values('t', axis=0, inplace=True)
    traces['t'] -= traces['t'].values[0]  # start time at 0
    traces.index = traces['t']  # move time into index
    traces.drop('t', axis=1, inplace=True)
    traces.columns = [str(vstep) for vstep in vsteps]  # set column names to vsteps
    return traces


def interpolate_traces(traces, dt):
    traces_interpolated = pd.DataFrame(index=np.arange(traces.index[0], traces.index[-1], dt))
    for i in traces.columns:
        not_nan = np.logical_not(traces[i].isnull().values)
        cs = CubicSpline(traces.index[not_nan], traces[i][not_nan])
        traces_interpolated[i] = cs(traces_interpolated.index.values)
    return traces_interpolated


def append_prepost_potential(traces, i_pre, i_post, dur_pre, dur_post, dt):
    time_pre = np.arange(0, dur_pre, dt)
    time_post = np.arange(0, dur_post, dt)
    pre_steps = pd.DataFrame(i_pre * np.ones((len(time_pre), len(traces.columns))), index=time_pre,
                             columns=traces.columns)
    post_steps = pd.DataFrame(i_post * np.ones((len(time_post), len(traces.columns))), index=time_post,
                             columns=traces.columns)

    traces.index += pre_steps.index[-1] + dt
    traces = pre_steps.append(traces)
    post_steps.index += traces.index[-1] + dt
    traces = traces.append(post_steps)
    return traces