import numpy as np
import pylab as pl
import pandas as pd
import csv

__author__ = 'caro'


def step_current2csv(cell_dir):
    """
    Transforms the step_current data file into a single csv file for each current step.
    """

    data = pd.read_csv(cell_dir + '/step_current/step_current.csv')
    steps = [-0.1, 0.1]
    header = np.array(['t', 'i', 'v', 'sec'], dtype=str)  # units: (ms), (nA), (mV)
    sec = np.zeros(np.array(data.t).size, dtype=object)  # section at which was recorded
    sec[0] = 'soma'

    # get all headers for membrane potential columns
    v_names = []
    for name in data.columns:
        if "v" in name:
            v_names.append(name)

    # unit conversion
    i_steps = np.array(data.i_steps) / 1000  # (nA)

    # get rid of nan values
    i_steps = i_steps[np.invert(np.isnan(i_steps))]

    # get membrane potential and current traces for the chosen current steps
    idx1 = np.argmin(np.abs(i_steps-steps[0]))
    idx2 = np.argmin(np.abs(i_steps-steps[1]))+1
    v_names = v_names[idx1:idx2]
    i_steps = i_steps[idx1:idx2]

    for idx, name in enumerate(v_names):
        # generate current trace
        i = np.array(data.i)
        i[i != 0] = 1  # where there was a current put 1
        i = i * i_steps[idx]  # and multiply by the amplitude of the current step

        # unit conversion
        t = np.array(data.t) * 1000
        v = np.array(data[name]) * 1000

        # save new csv
        data_new = np.column_stack((t, i, v, sec))
        with open(cell_dir + '/step_current/step_current_' + str(i_steps[idx]) + '.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            writer.writerow(header)
            writer.writerows(data_new)

if __name__ == "__main__":
    step_current2csv('./cell_2013_12_13f')
