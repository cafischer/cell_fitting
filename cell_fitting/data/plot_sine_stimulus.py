from __future__ import division
from cell_fitting.read_heka import get_v_and_t_from_heka, get_protocols_same_base
from cell_fitting.data import shift_v_rest
import os
import matplotlib.pyplot as pl
import numpy as np
import re


if __name__ == '__main__':
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol_base = 'Stimulus'
    v_rest_shift = -16
    correct_vrest = True
    dt = 0.05
    reg_exp_protocol = re.compile(protocol_base+'\([0-9]+\)')

    cells = ['2015_08_10g.dat', '2015_08_25b.dat', '2015_05_29h.dat', '2015_05_29f.dat', '2015_08_26b.dat',
             '2015_08_20i.dat', '2015_08_20d.dat', '2015_05_22r.dat', '2015_08_10e.dat', '2015_08_11d.dat',
             '2015_08_06c.dat', '2015_08_25d.dat', '2015_08_27c.dat', '2015_05_26e.dat', '2015_05_26a.dat',
             '2015_05_21o.dat', '2015_08_26e.dat', '2015_08_06a.dat', '2015_05_21m.dat', '2015_08_21b.dat',
             '2015_05_28e.dat', '2015_05_22s.dat', '2015_05_21n.dat', '2015_08_20b.dat', '2015_05_29c.dat',
             '2015_08_20c.dat', '2015_05_29e.dat', '2015_08_11c.dat', '2015_05_28c.dat', '2015_08_27e.dat',
             '2015_08_05b.dat', '2015_08_21f.dat', '2015_08_21e.dat', '2015_05_26f.dat', '2015_05_21p.dat',
             '2015_05_28b.dat', '2015_08_10a.dat', '2015_08_10f.dat', '2015_08_04b.dat', '2015_05_29d.dat',
             '2015_08_10b.dat', '2015_05_28a.dat', '2015_08_27d.dat', '2015_08_25e.dat', '2015_05_29i.dat',
             '2015_05_29a.dat', '2015_08_21a.dat', '2015_08_04a.dat', '2015_08_10d.dat', '2015_08_26f.dat',
             '2015_08_20a.dat', '2015_08_20j.dat', '2015_08_06d.dat', '2015_08_27b.dat', '2015_05_28d.dat',
             '2015_08_25h.dat', '2015_08_20f.dat', '2015_08_05a.dat', '2015_08_25g.dat', '2015_05_28f.dat',
             '2015_05_29g.dat', '2015_06_19l.dat', '2015_08_11e.dat', '2015_08_11b.dat', '2015_05_22q.dat',
             '2015_08_20e.dat', '2015_08_05c.dat', '2015_08_04c.dat', '2015_05_22t.dat', '2015_08_21c.dat',
             '2015_05_26b.dat', '2015_08_11f.dat']

    cells = ['2015_08_20f.dat']  # 11d seems not to be same as labbook

    for cell in cells:
        protocols = get_protocols_same_base(os.path.join(data_dir, cell), protocol_base)
        print protocols
        vms_per_protocol = []
        ts_per_protocol = []
        for protocol in protocols:
            v_mat, t_mat = get_v_and_t_from_heka(os.path.join(data_dir, cell), protocol)
            t = np.array(t_mat[0])
            v = np.array(v_mat[0])
            v = shift_v_rest(v, v_rest_shift)
            vms_per_protocol.append(v)
            ts_per_protocol.append(t)

            pl.figure()
            pl.title(protocol)
            pl.plot(t, v)
            pl.show()