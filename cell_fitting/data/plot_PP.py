import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_protocols_same_base
from cell_fitting.data import shift_v_rest
pl.style.use('paper')


save_dir = './plots'
data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
protocol = 'PP'
v_rest_shift = -16
#cells = get_cells_for_protocol(data_dir, protocol)
protocol_idx = 0

cells = ['2015_08_04d', '2015_08_05a', '2015_08_05b', '2015_08_05c', '2015_08_06d', '2015_08_10a', '2015_08_11d',
         '2015_08_11e', '2015_08_11f']
#cells = ['2015_08_10a']
offset = [2, 0, 2, 0, 0, 10, 10, 1, 0]
step_flags = [0, 1, 2]

for c_idx, cell in enumerate(cells):
    if not '2015' in cell:
        continue
    print cell
    file_dir = os.path.join(data_dir, cell+'.dat')

    for step_flag in step_flags:
        for seq in range(20):
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
                start = (seq * 30) + step_flag + offset[c_idx]
                end = min(len(protocols), (((seq+1) * 30)-2) + step_flag + offset[c_idx])
                for i in np.arange(start, end, 3):  # 10 for one run through  # 0-27+1, 30-57+1, 60-87+1  (+30 next range)
                    v_mat, t_mat = get_v_and_t_from_heka(file_dir, protocol if i == 0 else protocol+'('+str(i)+')')
                    t = np.array(t_mat[0])
                    v = np.array(v_mat[0])
                    v = shift_v_rest(v, v_rest_shift)

                    pl.plot(t, v, 'k', label='Exp. Data' if i == start else '')
                pl.ylabel('Membrane Potential (mV)')
                pl.xlabel('Time (ms)')
                pl.legend()
                pl.tight_layout()
                pl.savefig(os.path.join(save_dir_cell, 'PP' + str(seq) + '.png'))
                #pl.show()

                pl.figure()
                start = (seq * 30) + step_flag + offset[c_idx]
                end = min(len(protocols), (((seq + 1) * 30) - 2) + step_flag + offset[c_idx])
                for i in np.arange(start, end,
                                   3):  # 10 for one run through  # 0-27+1, 30-57+1, 60-87+1  (+30 next range)
                    v_mat, t_mat = get_v_and_t_from_heka(file_dir,
                                                         protocol if i == 0 else protocol + '(' + str(i) + ')')
                    t = np.array(t_mat[0])
                    v = np.array(v_mat[0])
                    v = shift_v_rest(v, v_rest_shift)

                    pl.plot(t, v, 'k', label='Exp. Data' if i == start else '')
                pl.ylabel('Membrane Potential (mV)')
                pl.xlabel('Time (ms)')
                if cell in ['2015_08_04d', '2015_08_05a']:
                    pl.xlim(265, 300)
                else:
                    pl.xlim(485, 560)
                pl.legend()
                pl.tight_layout()
                pl.savefig(os.path.join(save_dir_cell, 'PP' + str(seq) + '_zoom.png'))
                #pl.show()
            except KeyError:
                print i
                break