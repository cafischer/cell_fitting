import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.read_heka import get_sweep_index_for_amp, get_i_inj_from_function
from nrn_wrapper import Cell
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    #model_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/cells'
    #mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/hodgkinhuxley'

    # load model
    cell = Cell.from_modeldir(os.path.join(model_dir, 'hhCell.json'), mechanism_dir)

    # get trace of injected current
    protocol = 'rampIV'
    ramp_amp = 3.0
    tstop = 160
    dt = 0.01
    sweep_idx = get_sweep_index_for_amp(ramp_amp, protocol)
    i_inj = get_i_inj_from_function(protocol, [sweep_idx], tstop, dt)[0]

    # simulate
    simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': -75, 'tstop': tstop,
                         'dt': dt, 'celsius': 35, 'onset': 200}
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    # plot
    pl.figure()
    pl.plot(t, v, 'k')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.tight_layout()
    pl.show()