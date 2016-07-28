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

    # get indices for the chosen current steps
    indices = [np.where(i_steps == step)[0][0] for step in steps]

    f, (ax1, ax2) = pl.subplots(2, 1, sharex=True)
    for idx in indices:
        # generate current trace
        i_amp = np.array(data.i)
        i_amp[i_amp != 0] = 1  # where there was a current put 1
        i_amp = i_amp * i_steps[idx]  # and multiply by the amplitude of the current step
        i['i'] = i_amp

        # find the start V
        v_start = np.mean(data[[v_names[0]]][v_names[0]][:np.where(i_amp==i_steps[idx])[0][0]] * 1000)

        # unit conversion
        t = data[['t']] * 1000  # convert unit to ms
        v = data[[v_names[idx]]] * 1000  # convert unit to mV

        # shift the whole membrane potential according to the first measured trace
        v_new = v + v_start - v[v_names[idx]][0]

        # save to .csv
        data_new = pd.concat([t, i, v_new, sec], axis=1)
        newfilename = '{}/stepcurrent/stepcurrent_shifted{}.csv'.format(cell_dir, i_steps[idx])
        data_new.columns = header
        #data_new.to_csv(newfilename)

        # plot data
        f, (ax1, ax2) = pl.subplots(2, 1, sharex=True)
        ax1.plot(data_new.as_matrix(['t']), data_new.as_matrix(['v']), 'k')
        #ax1.plot(data_new.as_matrix(['t']), v_new, 'k')
        ax2.plot(data_new.as_matrix(['t']), data_new.as_matrix(['i']), 'k')
        ax2.set_xlabel('Time (ms)', fontsize=18)
        ax1.set_ylabel('Membrane \npotential (mV)', fontsize=18)
        ax2.set_ylabel('Current (nA)', fontsize=18)
        ax2.set_ylim(-0.12, 0.02)
        pl.savefig(cell_dir + '/stepcurrent/stepcurrent_'+str(i_steps[idx]*1000)+'.png')
    #pl.savefig(cell_dir + '/stepcurrent/stepcurrent_shifted.png')
    pl.show()


if __name__ == "__main__":
    step_current2csv('./new_cells/2015_08_11d', [-0.1])
