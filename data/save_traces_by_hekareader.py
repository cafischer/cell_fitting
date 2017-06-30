from heka_reader import HekaReader
import pandas as pd
import os
import re
import matplotlib.pyplot as pl
import numpy as np


if __name__ == '__main__':

    cell = '2015_08_06d'
    file_dir = './'+cell+'/'+cell +'.dat'
    #file_dir = os.path.join('/home/caro/Downloads/rawData', cell)
    vrest = -75
    v_rest_change = None #-16
    correct_vrest = True

    hekareader = HekaReader(file_dir)
    type_to_index = hekareader.get_type_to_index()

    for w in range(1, 2):
        group = 'Group1'
        protocol = 'PP'  #'PP('+str(w)+')'
        trace = 'Trace1'
        protocol_to_series = hekareader.get_protocol(group)
        series = protocol_to_series[protocol]
        sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series])+1)]
        print '# sweeps: ', len(sweeps)
        #sweep_idx = [0]
        sweep_idx = range(len(sweeps))
        #sweep_idx = [range(len(sweeps))[-1]]
        sweeps = [sweeps[index] for index in sweep_idx]

        indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]

        fig = pl.figure()
        ax = fig.add_subplot(111)
        for i, index in enumerate(indices):
            # plot
            x, y = hekareader.get_xy(index)
            x *= 1000
            y *= 1000
            if correct_vrest:
                if vrest is not None:
                    y = y - (y[0] - vrest)
                if v_rest_change is not None:
                    y += v_rest_change
            x_unit, y_unit = hekareader.get_units_xy(index)

            ax.plot(x, y)
            ax.set_xlabel('Time (ms)', fontsize=18)
            ax.set_ylabel('Membrane Potential (mV)', fontsize=18)
            ax.tick_params(labelsize=15)

            # save data
            protocol_tmp = re.sub('\(.*\)', '', protocol)
            #if protocol_tmp == 'PP(0)':
            try:
                i_inj = pd.read_csv('./Protocols/' + protocol + '.csv', header=None)  # TODO: different for all PPs
                i_inj = np.array(i_inj)[:, 0]
            except IOError:
                print 'Using different current protocol!'
                i_inj = pd.read_csv('./Protocols/' + 'PP(3)' + '.csv', header=None)  # TODO: different for all PPs
                i_inj = np.array(i_inj)[:, 0]

            if protocol == 'IV':
                amp = -0.15 + sweep_idx[i] * 0.05
                amp_change = amp
            elif protocol == 'rampIV':
                amp = sweep_idx[i] * 0.1
                amp_change = amp / 0.1
            elif protocol == 'hypTester':
                amp = -0.005
                amp_change = 1
            elif protocol == 'Zap20':
                amp = 0.1
                amp_change = 1
            else:
                amp = 0
                amp_change = 1
            print 'Amplitude: ', amp
            i_inj *= amp_change

            data = pd.DataFrame({'v': y, 't': x}) #, 'i': i_inj})
            save_dir = os.path.join('./', cell, 'PP_no_inj', protocol)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            #shorten = np.logical_and(470 <= x, x <= 650)  TODO
            #data = data[shorten].copy()
            #data.t -= data.t.iloc[0]

            data.to_csv(save_dir + '/' + str(amp) + '(nA).csv', index=False)
        #ax.set_xlim([0, 120])
        #ax.set_ylim([-70, 55])
        pl.tight_layout()
        #pl.show()