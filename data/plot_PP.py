import matplotlib.pyplot as pl
import pandas as pd
import os

for seq in range(20):

    step = 1
    if step == 0:
        step_str = 'step0nA'
    elif step == 1:
        step_str = 'step-0.1nA'
    elif step == 2:
        step_str = 'step0.1nA'

    save_dir = os.path.join('2014_03_18e/PP_no_inj/img/', step_str)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pl.figure()
    pl.title('1st Ramp = 4 nA, 2nd Ramp = '+str(seq*0.05+1.8)+' nA')
    for i in range((seq * 30) + step, (((seq+1) * 30)-2) + step, 3):  # 10 for one run through  # 0-27+1, 30-57+1, 60-87+1  (+30 next range)
        data = pd.read_csv('2015_08_06d/PP_no_inj/PP('+str(i)+')/0(nA).csv')
        pl.plot(data.t, data.v, 'k', label='Exp.Data' if i==seq*30 else '')
    pl.ylabel('Membrane Potential (mV)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.xlim(485, 560)
    pl.legend(fontsize=16)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'PP'+str(seq)+'.png'))
    pl.show()