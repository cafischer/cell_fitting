import numpy as np
import pylab as pl
import pandas as pd
import csv
import copy

__author__ = 'caro'


def dap2csv(cell_dir):
    """
    Transforms the step_current data file into a single csv file for each current step.
    """

    # read data
    data = pd.read_csv(cell_dir + '/dap/DAP_hyperpolarization.csv').convert_objects(convert_numeric=True)
    header = ['t', 'i', 'v', 'sec']
    sec = pd.Series(dtype=str)

    for n in range(1, 10):
        i = data[['i'+str(n)]]
        t = data[['t']] * 1000  # convert unit to ms
        v = data[['v'+str(n)]] * 1000  # convert unit to mV

        # save to .csv
        data_new = pd.concat([t, i, v], axis=1)
        data_new['sec'] = sec
        data_new['sec'][0] = 'soma'
        newfilename = '{}/dap/dap_{}.csv'.format(cell_dir, n)
        data_new.columns = header
        data_new.to_csv(newfilename)

        # plot
        pl.figure()
        pl.plot(data_new.as_matrix(['t']), data_new.as_matrix(['v']), 'k', linewidth=2)
        #pl.plot(data_new.as_matrix(['t']), data_new.as_matrix(['i']))
        pl.show()

if __name__ == "__main__":
    dap2csv('./cell_2013_12_13f')
