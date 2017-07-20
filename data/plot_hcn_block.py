from heka_reader import HekaReader
import os
import matplotlib.pyplot as pl
from data import correct_baseline
import re


def get_indices(group, sweep_idx):
    type_to_index = hekareader.get_type_to_index()
    trace = 'Trace1'
    protocol_to_series = hekareader.get_protocol(group)
    if not protocol in protocol_to_series.keys():
        return None
    series = protocol_to_series[protocol]
    sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series]) + 1)]
    sweeps = [sweeps[index] for index in sweep_idx]
    indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]
    return indices


if __name__ == '__main__':

    cells = ['2015_08_20e.dat', '2015_08_21a.dat', '2015_08_21b.dat', '2015_08_21e.dat', '2015_08_21f.dat',
             '2015_08_26f.dat']  # there are probably more: see labbooks
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/rawData'
    v_rest = None
    correct_vrest = True
    protocol_base = 'IV'
    protocol = protocol_base
    dt = 0.05
    reg_exp_protocol = re.compile(protocol_base+'\([0-9]+\)')

    for cell in cells:
        hekareader = HekaReader(os.path.join(data_dir, cell))
        group = 'Group1'
        protocol_to_series = hekareader.get_protocol(group)
        n_protocols = sum([1 if reg_exp_protocol.match(p) else 0 for p in protocol_to_series])
        vms = []
        for i in [0, n_protocols-1]:
            if i == 0:
                protocol = protocol_base
            else:
                protocol = protocol_base+'('+str(i)+')'
            sweep_idx = [0]
            indices = get_indices(group, sweep_idx)
            if indices is None:
                continue
            else:
                index = indices[0]
            t, vm = hekareader.get_xy(index)
            t *= 1000  # ms
            vm *= 1000  # mV
            assert dt == t[1] - t[0]  # ms
            if correct_vrest:
                vm = correct_baseline(vm, v_rest)
            vms.append(vm)

        pl.figure()
        pl.title(cell)
        pl.plot(t, vms[0], 'k', label='before ZD')
        pl.plot(t, vms[1], 'b', label='ZD')
        pl.show()