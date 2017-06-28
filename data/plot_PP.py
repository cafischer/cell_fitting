import matplotlib.pyplot as pl
import pandas as pd


pl.figure()
for i in range(100, 120, 3):  #0, 28, 3
    data = pd.read_csv('2015_08_06d/PP_no_inj/PP('+str(i)+')/0(nA).csv')
    pl.plot(data.t, data.v+16, 'k', label='Exp.Data' if i==0 else '')
pl.ylabel('Membrane Potential (mV)', fontsize=16)
pl.xlabel('Time (ms)', fontsize=16)
pl.xlim(485, 560)
pl.legend(fontsize=16)
pl.tight_layout()
pl.show()