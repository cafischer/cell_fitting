import os
import matplotlib.pyplot as pl
from cell_fitting.read_heka import *
import pandas as pd


if __name__ == '__main__':
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol_dir = '../data/Protocols'
    cell_id = '2013_12_11a'  #'2015_08_06d'
    file_dir = os.path.join(data_dir, cell_id + '.dat')
    protocol = 'hyperRampTester'

    v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(file_dir, protocol, group='Group1', trace='Trace1',
                                                     sweep_idxs=None, return_sweep_idxs=True)
    tstop = t_mat[0][-1]
    dt = t_mat[0][1] - t_mat[0][0]

    # from function
    i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, tstop, dt)

    # from csv
    i_inj_csv = pd.read_csv(os.path.join(protocol_dir, protocol+'.csv')).values
    # for zap
    # i_inj_csv = np.loadtxt(os.path.join(protocol_dir, protocol+'.txt'))
    # i_frame = pd.DataFrame(i_inj_csv)
    # i_frame.to_csv(os.path.join(protocol_dir, protocol+'.csv'), index=False, header=False)

    for t, i_inj in zip(t_mat, i_inj_mat):
        pl.figure()
        pl.plot(np.arange(len(i_inj))*dt, i_inj, 'b', label='function')
        pl.plot(np.arange(len(i_inj_csv))*dt, i_inj_csv, 'r', label='csv')
        # for zap
        #pl.plot(np.arange(len(i_inj_csv)) * dt + 2000, i_inj_csv, 'r', label='csv')
        pl.legend(loc='upper right')
        pl.show()