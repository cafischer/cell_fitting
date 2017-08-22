import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol
from cell_fitting.data import shift_v_rest


save_dir = './plots'
data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
protocol = 'PP'
v_rest_shift = -16
cells = get_cells_for_protocol(data_dir, protocol)

for cell in cells:
    if not '2015' in cell:
        continue
    print cell
    file_dir = os.path.join(data_dir, cell)

    for seq in range(20):

        step_flag = 0
        if step_flag == 0:
            step_str = 'step0nA'
        elif step_flag == 1:
            step_str = 'step-0.1nA'
        elif step_flag == 2:
            step_str = 'step0.1nA'

        save_dir_cell = os.path.join(save_dir, 'PP_no_i_inj', cell, step_str)

        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)
        try:
            pl.figure()
            pl.title('1st Ramp = 4 nA, 2nd Ramp = '+str(seq*0.05+1.8)+' nA')
            for i in range((seq * 30) + step_flag, (((seq+1) * 30)-2) + step_flag, 3):  # 10 for one run through  # 0-27+1, 30-57+1, 60-87+1  (+30 next range)
                v_mat, t_mat = get_v_and_t_from_heka(file_dir, protocol if i == 0 else protocol+'('+str(i)+')')
                t = np.array(t_mat[0])
                v = np.array(v_mat[0])
                v = shift_v_rest(v, v_rest_shift)

                pl.plot(t, v, 'k', label='Exp.Data' if i==seq*30 else '')
            pl.ylabel('Membrane Potential (mV)', fontsize=16)
            pl.xlabel('Time (ms)', fontsize=16)
            #pl.xlim(485, 560)
            pl.legend(fontsize=16)
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_cell, 'PP' + str(seq) + '_no_zoom.png'))
            # pl.show()
        except KeyError:
            break