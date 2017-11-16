import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_protocols_same_base
from cell_fitting.data import shift_v_rest
pl.style.use('paper')


save_dir = './plots'
data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
protocol = 'PP_tester'
v_rest_shift = -16
protocol_idx = 0
sweep_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#cells = ['2015_08_04d', '2015_08_05a', '2015_08_05b', '2015_08_05c', '2015_08_06d', '2015_08_10a', '2015_08_11d','2015_08_11e', '2015_08_11f']
cells = ['2015_08_11f']

for cell in cells:
    if not '2015' in cell:
        continue
    print cell
    file_dir = os.path.join(data_dir, cell+'.dat')

    for sweep_idx in sweep_idxs:

        step_flag = 0
        if step_flag == 0:
            step_str = 'step0nA'
        elif step_flag == 1:
            step_str = 'step-0.1nA'
        elif step_flag == 2:
            step_str = 'step0.1nA'

        save_dir_cell = os.path.join(save_dir, 'PP', cell, step_str)

        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        protocols = get_protocols_same_base(file_dir, protocol)
        try:
            pl.figure()
            for i in range(0, len(protocols), 1):
                v_mat, t_mat = get_v_and_t_from_heka(file_dir, protocol if i == 0 else protocol+'('+str(i)+')',
                                                     sweep_idxs=[sweep_idx])
                t = np.array(t_mat[0])
                v = np.array(v_mat[0])
                v = shift_v_rest(v, v_rest_shift)

                pl.plot(t, v, 'k', label='Exp. Data sweep: '+ str(sweep_idx) if i == 0 else '')
            pl.ylabel('Membrane Potential (mV)', fontsize=16)
            pl.xlabel('Time (ms)', fontsize=16)
            #pl.xlim(485, 560)
            pl.legend(fontsize=16)
            pl.tight_layout()
            pl.show()
        except KeyError:
            break