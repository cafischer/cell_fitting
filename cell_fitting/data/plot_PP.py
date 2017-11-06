from __future__ import division
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
protocol_idx = 14

#cells = get_cells_for_protocol(data_dir, protocol)
#offset = np.zeros(len(cells))
#cells = ['2014_07_08e']
#offset = [0]  # 2014_03_18b: 43; 2014_03_18f: 47; 2014_03_14i: 116

#cells_short = ['2015_08_04d', '2015_08_05a']  # offset_short = [2, 0]
#cells = ['2015_08_05b', '2015_08_05c', '2015_08_06d', '2015_08_10a', '2015_08_11d',
#         '2015_08_11e', '2015_08_11f']
#offset = [2, 0, 0, 10, 10, 62, 0]  # 11e 10 und 62 offset
cells = ['2015_08_11e']
offset = [1]
step_flags = [0, 1, 2]
len_ramp3_times = 10  # TODO: 10 for 2015  # 12 for early 2014
len_t = 69200  #46900  #93200  #69200

for c_idx, cell in enumerate(cells):
    #if not '2014' in cell:
    #    continue
    print cell
    file_dir = os.path.join(data_dir, cell+'.dat')

    for step_flag in step_flags:
        if step_flag == 0:
            step_amp = 0
        elif step_flag == 1:
            step_amp = -0.1
        elif step_flag == 2:
            step_amp = 0.1
        step_str = 'step_%.1f(nA)' % step_amp

        save_dir_cell = os.path.join(save_dir, 'PP', cell, step_str)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        protocols = get_protocols_same_base(file_dir, protocol)
        ramp3_amp_idxs = max(1, int(np.floor(len(protocols) / (3 * len_ramp3_times))))

        v_mat = np.zeros((ramp3_amp_idxs, len_ramp3_times, len_t))
        for ramp3_amp_idx in range(ramp3_amp_idxs):
            start = int((ramp3_amp_idx * 3 * len_ramp3_times) + step_flag + offset[c_idx])
            end = int(min(len(protocols), (((ramp3_amp_idx + 1) * (3 * len_ramp3_times)) - 2) + step_flag + offset[c_idx]))
            ramp3_time_protocols = range(start, end, 3)

            for ramp3time_idx, ramp3_time_protocol in enumerate(ramp3_time_protocols):
                v_tmp, t_tmp = get_v_and_t_from_heka(file_dir, protocol if ramp3_time_protocol == 0
                                                     else protocol + '(' + str(ramp3_time_protocol) + ')')
                t = np.array(t_tmp[0])[:len_t]
                v = np.array(v_tmp[0])[:len_t]
                v = shift_v_rest(v, v_rest_shift)
                v_mat[ramp3_amp_idx, ramp3time_idx, :] = v

                # pl.figure()
                # pl.plot(t, v)
                # pl.xlim(700, 800)
                # pl.show()

            pl.figure()
            for ramp3time_idx in range(len(ramp3_time_protocols)):
                pl.plot(t, v_mat[ramp3_amp_idx, ramp3time_idx, :], 'k', label='Exp. Data' if ramp3time_idx == 0 else '')
            pl.ylabel('Membrane Potential (mV)')
            pl.xlabel('Time (ms)')
            pl.legend()
            pl.tight_layout()
            #pl.savefig(os.path.join(save_dir_cell, 'PP' + str(ramp3_amp_idx) + '.png'))
            #pl.show()

            pl.figure()
            for ramp3time_idx in range(len(ramp3_time_protocols)):
                pl.plot(t, v_mat[ramp3_amp_idx, ramp3time_idx, :], 'k', label='Exp. Data' if ramp3time_idx == 0 else '')
            pl.ylabel('Membrane Potential (mV)')
            pl.xlabel('Time (ms)')
            if cell in ['2015_08_04d', '2015_08_05a']:
                pl.xlim(265, 300)
            elif '2014' in cell:
                pl.xlim(725, 760)
            else:
                pl.xlim(485, 560)
            pl.legend()
            pl.tight_layout()
            #pl.savefig(os.path.join(save_dir_cell, 'PP' + str(ramp3_amp_idx) + '_zoom.png'))
            pl.show()

        #np.save(os.path.join(save_dir_cell, 'v_mat.npy'), v_mat)
        #np.save(os.path.join(save_dir_cell, 't.npy'), t)