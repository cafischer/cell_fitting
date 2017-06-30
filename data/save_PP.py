from heka_reader import HekaReader
import pandas as pd
import os
import re
import matplotlib.pyplot as pl
import numpy as np


if __name__ == '__main__':

    cell = '2015_08_06d'
    file_dir = './'+cell+'/'+cell +'.dat'
    #file_dir = os.path.join('/home/cf/Phd/DAP-Project/cell_data/rawData', cell)
    vrest = -75
    v_rest_change = None
    correct_vrest = True

    hekareader = HekaReader(file_dir)
    type_to_index = hekareader.get_type_to_index()

    for w in range(1, 2):
        group = 'Group1'
        if w == 0:
            protocol == 'PP'
        else:
            protocol = 'PP('+str(w)+')'
        trace = 'Trace1'
        protocol_to_series = hekareader.get_protocol(group)
        series = protocol_to_series[protocol]
        sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series])+1)]
        print '# sweeps: ', len(sweeps)
        sweep_idx = range(len(sweeps))
        sweeps = [sweeps[index] for index in sweep_idx]

        indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]

        #fig = pl.figure()
        #ax = fig.add_subplot(111)
        for i, index in enumerate(indices):
            x, y = hekareader.get_xy(index)
            x *= 1000
            y *= 1000
            if correct_vrest:
                if vrest is not None:
                    y = y - (y[0] - vrest)
                if v_rest_change is not None:
                    y += v_rest_change
            x_unit, y_unit = hekareader.get_units_xy(index)

            #ax.plot(x, y)
            #ax.set_xlabel('Time (ms)', fontsize=18)
            #ax.set_ylabel('Membrane Potential (mV)', fontsize=18)
            #ax.tick_params(labelsize=15)

            # save data
            data = pd.DataFrame({'v': y, 't': x})
            if protocol == 'PP':
                protocol = 'PP(0)'  # easier for later use
            save_dir = os.path.join('./', cell, 'PP_no_inj', protocol)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            data.to_csv(os.path.join(save_dir, '0(nA).csv'), index=False)
        #ax.set_xlim([0, 120])
        #ax.set_ylim([-70, 55])
        #pl.tight_layout()
        #pl.show()