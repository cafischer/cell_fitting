from __future__ import division
from heka_reader import HekaReader
import os
import matplotlib.pyplot as pl
import re
import numpy as np
from cell_fitting.data import shift_v_rest


if __name__ == '__main__':
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol_base = 'Stimulus'
    v_rest = -75
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

    for cell in cells:
        hekareader = HekaReader(os.path.join(data_dir, cell))
        type_to_index = hekareader.get_type_to_index()
        group = 'Group1'
        trace = 'Trace1'
        protocol_to_series = hekareader.get_protocol(group)

        vms_per_protocol = []
        ts_per_protocol = []
        n_protocols = sum([1 if reg_exp_protocol.match(p) else 0 for p in protocol_to_series])
        for i in [0, n_protocols - 1]:
            if i == 0:
                protocol = protocol_base
            else:
                protocol = protocol_base + '(' + str(i) + ')'
            if not protocol in protocol_to_series.keys():
                continue
            series = protocol_to_series[protocol]
            sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series]) + 1)]
            sweep_idx = range(len(sweeps))
            sweeps = [sweeps[index] for index in sweep_idx]
            indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]

            vms = []
            ts = []
            for index in indices:
                # take next sweep
                t, vm = hekareader.get_xy(index)
                t *= 1000  # ms
                assert dt == t[1] - t[0]  # ms
                vm *= 1000  # mV
                if correct_vrest:
                    vm = shift_v_rest(vm, v_rest)
                vms.append(vm)
                ts.append(t)
            vms_per_protocol.append(vms)
            ts_per_protocol.append(ts)

        for ts, vms in zip(ts_per_protocol, vms_per_protocol):
            pl.figure()
            pl.title(cell)
            for t, vm in zip(ts, vms):
                pl.plot(t, vm)
            pl.show()