import os
import matplotlib.pyplot as pl
from cell_fitting.data import shift_v_rest
from cell_fitting.read_heka import get_v_and_t_from_heka, get_protocols_same_base
import re
import numpy as np
pl.style.use('paper')


if __name__ == '__main__':

    save_dir = 'plots/hcn_block'
    cells = ['2015_08_20e.dat', '2015_08_21e.dat', '2015_08_21f.dat',
             '2015_08_26f.dat']  # there are probably more: see labbooks
    cells = ['2015_08_21a.dat', '2015_08_21b.dat']
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    v_rest = None
    v_shift = -16
    protocol_base = 'plot_IV' #plot_IV  #Zap20 #rampIV
    protocol = protocol_base
    reg_exp_protocol = re.compile(protocol_base+'(\([0-9]+\))?')
    save_dir = os.path.join(save_dir, protocol)

    for cell in cells:
        protocols = get_protocols_same_base(os.path.join(data_dir, cell), protocol)

        v_mat, t_mat = get_v_and_t_from_heka(os.path.join(data_dir, cell), protocols[0])
        v_before = shift_v_rest(v_mat[1], v_shift)
        v_mat, t_mat = get_v_and_t_from_heka(os.path.join(data_dir, cell), protocols[-1])
        v_after = shift_v_rest(v_mat[1], v_shift)
        t = t_mat[0]

        # plot
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        pl.figure()
        #pl.title(cell, fontsize=16)
        pl.plot(t, v_before, 'k', label='before ZD')
        pl.plot(t, v_after, 'k', label='after ZD', alpha=0.5)
        st = np.ceil(pl.ylim()[1] / 5) * 5
        pl.yticks(np.arange(st, st + 11 * -5, -5))
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane potential (mV)')
        pl.legend(loc='lower right')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir, cell[:-3]+'png'))
        pl.show()