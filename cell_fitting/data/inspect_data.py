import pandas as pd
import matplotlib.pyplot as pl

__author__ = 'caro'

cell = '2015_05_28c'
data_dirs = [   #'./2015_08_26b/vrest-75/rampIV/3.0(nA).csv',
                #'./2015_08_06d/vrest-75/PP(0)(3)/0(nA).csv',
                #'./'+cell+'/vrest-80/rampIV/3.3(nA).csv',
                './'+cell+'/vrest-80/IV/-0.1(nA).csv',
                './'+cell+'/vrest-80/IV/0.2(nA).csv',
                './'+cell+'/vrest-80/IV/0.5(nA).csv'
             ]

data_sets = list()
for i, data_dir in enumerate(data_dirs):
    data_sets.append(pd.read_csv(data_dir))

fig, ax = pl.subplots(2, 1, sharex=True)
for data in data_sets:
    ax[0].plot(data.t, data.v, 'k')
    ax[1].plot(data.t, data.i, 'k')
    ax[0].set_ylabel('Membrane Potential (mV)')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Current (nA)')
#ax[0].set_xlim([0, 80])
#ax[1].set_xlim([0, 80])
pl.tight_layout()
pl.show()

ax = pl.subplot()
for data in data_sets:
    ax.plot(data.t, data.v, 'k')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_xlabel('Time (ms)')
pl.tight_layout()
pl.show()
