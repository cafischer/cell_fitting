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
cells = ['2014_07_03e']
    #, '2014_03_18e', '2014_03_14g', '2014_07_03e', '2014_03_18b','2014_03_18d', '2014_03_19b',
     #    '2014_03_19e', '2014_03_14d', '2014_03_18f','2014_07_02e','2014_07_08e', '2014_03_14b']
offset = [0]  # '2014_03_14i': 116 still looks weird, not 10 bumps?
len_t = 93200  #46900  #93200  #69200

for c_idx, cell in enumerate(cells):
    print cell
    file_dir = os.path.join(data_dir, cell+'.dat')
    protocols = get_protocols_same_base(file_dir, protocol)

    start = 10
    end = start + 3
    ramp3_time_protocols = range(start, end, 1)
    v_mat = np.zeros((len(ramp3_time_protocols), len_t))

    for ramp3time_idx, ramp3_time_protocol in enumerate(ramp3_time_protocols):
        v_tmp, t_tmp = get_v_and_t_from_heka(file_dir, protocol if ramp3_time_protocol == 0
                                             else protocol + '(' + str(ramp3_time_protocol) + ')')
        t = np.array(t_tmp[0])[:len_t]
        v = np.array(v_tmp[0])[:len_t]
        v = shift_v_rest(v, v_rest_shift)
        v_mat[ramp3time_idx, :] = v

    pl.figure()
    for ramp3time_idx in range(len(ramp3_time_protocols)):
        pl.plot(t, v_mat[ramp3time_idx, :], 'k', label='Exp. Data' if ramp3time_idx == 0 else '')
    pl.ylabel('Membrane Potential (mV)')
    pl.xlabel('Time (ms)')
    pl.legend()
    pl.tight_layout()

    pl.figure()
    for ramp3time_idx in range(len(ramp3_time_protocols)):
        pl.plot(t, v_mat[ramp3time_idx, :], 'k', label='Exp. Data' if ramp3time_idx == 0 else '')
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
    pl.show()