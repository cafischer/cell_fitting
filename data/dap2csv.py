from __future__ import division
import pylab as pl
import pandas as pd
import csv
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

__author__ = 'caro'


def dap2csv(cell_dir):
    """
    Transforms the step_current data file into a single csv file for each current step.
    """

    # read data
    data = pd.read_csv(cell_dir + '/dap/dap.csv').convert_objects(convert_numeric=True)
    header = ['t', 'i', 'v', 'sec']
    sec = pd.Series(dtype=str)

    for n in range(1):
        i = data[['i']]
        t = data[['t']] # * 1000  # convert unit to ms
        v = data[['v']] #* 1000  # convert unit to mV

        # TODO
        #import numpy as np
        #v1 = np.array(v)
        #dt = 0.01
        #v1[10.0/dt:13.5/dt] = v1[10.0/dt]
        #v.v = v1
        # TODO

        # save to .csv
        data_new = pd.concat([t, i, v], axis=1)
        data_new['sec'] = sec
        data_new['sec'][0] = 'soma'
        newfilename = '{}/dap/dap_withoutAP{}.csv'.format(cell_dir, n)  # TODO
        data_new.columns = header
        data_new.to_csv(newfilename)

        # plot
        f, (ax1, ax2) = pl.subplots(2, 1, sharex=True)
        ax1.plot(data_new.as_matrix(['t']), data_new.as_matrix(['v']), 'k', linewidth=2)
        ax2.plot(data_new.as_matrix(['t']), data_new.as_matrix(['i']), 'k', linewidth=2)
        ax2.set_xlabel('Time (ms)', fontsize=18)
        ax1.set_ylabel('Membrane \npotential (mV)', fontsize=18)
        ax2.set_ylabel('Current (nA)', fontsize=18)
        pl.savefig(cell_dir + '/dap/dap.png')
        pl.show()

        pl.figure()
        pl.plot(data_new.as_matrix(['t']), data_new.as_matrix(['v']), 'k', linewidth=2)
        pl.xlabel('Time (ms)', fontsize=18)
        pl.ylabel('Membrane \npotential (mV)', fontsize=18)
        pl.xlim([10, 60])
        pl.savefig(cell_dir + '/dap/daponly.png')
        pl.show()

        #f, (ax1) = pl.subplots(1, 1, sharex=True)
        #ax1.plot(data_new.as_matrix(['t']), data_new.as_matrix(['v']), 'k', linewidth=2)
        #ax1.set_xlabel('Time (ms)', fontsize=18)
        #ax1.set_ylabel('Membrane \npotential (mV)', fontsize=18)
        #pl.savefig(cell_dir + '/dap/dap2.png')
        #pl.show()

if __name__ == "__main__":
    dap2csv('./new_cells/2015_08_11d')
