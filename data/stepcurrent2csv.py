import numpy as np
import pylab as pl
import pandas as pd
import csv

__author__ = 'caro'


def step_current2csv(cell_dir, steps):
    """
    Transforms the step_current data file into a single csv file for each current step.
    """

    data = pd.read_csv(cell_dir + '/stepcurrent/stepcurrent.csv').convert_objects(convert_numeric=True)
    header = ['t', 'i', 'v', 'sec']  # units: (ms), (nA), (mV)
    sec = pd.DataFrame()
    section = pd.Series(data=['soma'], dtype=str)
    sec['sec'] = section
    i = pd.DataFrame()

    # get all headers for membrane potential columns
    v_names = []
    for name in data.columns:
        if "v" in name:
            v_names.append(name)

    i_steps = np.array(data.i_steps) / 1000  # covert unit to nA
    i_steps = i_steps[np.invert(np.isnan(i_steps))]  # get rid of nan values

    # get membrane potential and current traces for the chosen current steps
    idx1 = np.argmin(np.abs(i_steps-steps[0]))
    idx2 = np.argmin(np.abs(i_steps-steps[1]))+1
    v_names = v_names[idx1:idx2]
    i_steps = i_steps[idx1:idx2]

    for idx, name in enumerate(v_names):
        # generate current trace
        i_amp = np.array(data.i)
        i_amp[i_amp != 0] = 1  # where there was a current put 1
        i_amp = i_amp * i_steps[idx]  # and multiply by the amplitude of the current step
        i['i'] = i_amp

        # unit conversion
        t = data[['t']] * 1000  # convert unit to ms
        v = data[[name]] * 1000  # convert unit to mV

        # save to .csv
        data_new = pd.concat([t, i, v, sec], axis=1)
        newfilename = '{}/stepcurrent/stepcurrent{}.csv'.format(cell_dir, i_steps[idx])
        data_new.columns = header
        data_new.to_csv(newfilename)


if __name__ == "__main__":
    step_current2csv('./cell_2013_12_13f', [-0.1, 0.1])
