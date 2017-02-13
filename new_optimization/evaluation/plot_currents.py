import json
from new_optimization.fitter.hodgkinhuxleyfitter import *
import matplotlib.pyplot as pl
import numpy as np
from optimization.helpers import *
from evaluate import *


def plot_currents(save_dir, candidate, data_dir):

    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    optimization_settings['fitter']['data_dir'] = data_dir
    optimization_settings['fitter']['mechanism_dir'] = None
    fitter = HodgkinHuxleyFitter(**optimization_settings['fitter'])
    # TODO
    #fitter.cell.insert_mechanisms([[['soma', '0.5', 'ih_fast', 'gbar']]])
    #fitter.cell.insert_mechanisms([[['soma', '0.5', 'ih_fast', 'gbar']], [['soma', '0.5', 'ih_slow', 'gbar']]])
    #fitter.cell.update_attr(['soma', '0.5', 'ih_fast', 'gbar'], 0.0001)
    #fitter.cell.update_attr(['soma', '0.5', 'ih_slow', 'gbar'], 0.001)

    channel_list = get_channel_list(fitter.cell, 'soma')
    ion_list = get_ionlist(channel_list)

    # record currents
    currents = np.zeros(len(channel_list), dtype=object)
    for i in range(len(channel_list)):
        currents[i] = fitter.cell.soma.record_from(channel_list[i], 'i' + ion_list[i], pos=.5)

    # apply vclamp
    v_model, t, i_inj = fitter.simulate_cell(candidate)

    # convert current traces to array
    for i in range(len(channel_list)):
        if 'onset' in fitter.simulation_params:
            real_start = int(round(fitter.simulation_params['onset'] / fitter.simulation_params['dt']))
            currents[i] = np.array(currents[i])[real_start:]
        currents[i] = np.array(currents[i])

    # plot current traces
    pl.figure()
    for i in range(len(channel_list)):
        pl.plot(t, -1 * currents[i], label=channel_list[i])
        pl.ylabel('Current (mA/cm2)', fontsize=16)
        pl.xlabel('Time (ms)', fontsize=16)
        pl.legend(fontsize=16)
    pl.show()


if __name__ == '__main__':
    save_dir = '../../results/new_optimization/2015_08_26b/01_02_17_readjust_newih0/'
    method = 'L-BFGS-B'
    n_best = 0
    data_dir = '../../data/2015_08_26b/corrected_vrest2/rampIV/3.0(nA).csv'
    #data_dir = '../../data/2015_08_26b/corrected_vrest2/IV/-0.15(nA).csv'

    best_candidate = plot_best_candidate(save_dir+method+'/', n_best)
    plot_currents(save_dir+method+'/', best_candidate, data_dir)