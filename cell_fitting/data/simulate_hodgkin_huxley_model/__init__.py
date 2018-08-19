import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.read_heka import get_i_inj_from_function, get_sweep_index_for_amp


if __name__ == '__main__':
    model_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/cells/hhCell.json'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/hodgkinhuxley'
    data_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/toymodels/hhCell/'
    protocol = 'rampIV'
    amp = 3.0

    data_dir = os.path.join(data_dir, protocol, 'amp_'+str(amp))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # simulation parameters
    tstop = 161.99
    dt = 0.01
    i_inj = get_i_inj_from_function(protocol, [get_sweep_index_for_amp(amp, protocol)], tstop, dt)[0]
    simulation_params =  {'i_inj': i_inj, 'v_init': -75, 'tstop': tstop, 'dt': dt, 'pos_i': 0.5,
                          'pos_v': 0.5, 'sec': ('soma', None), 'celsius': 6.3, 'onset': 200}

    # simulate
    v, t, i_inj = iclamp_handling_onset(cell, **simulation_params)

    # save
    np.save(os.path.join(data_dir, 'v.npy'), v)
    np.save(os.path.join(data_dir, 't.npy'), t)
    np.save(os.path.join(data_dir, 'i_inj.npy'), i_inj)

    # plot
    pl.figure()
    pl.plot(t, v, 'k')
    pl.show()