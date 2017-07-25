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

    save_dir = 'plots/hcn_block'
    cells = ['2015_08_20e.dat', '2015_08_21a.dat', '2015_08_21b.dat', '2015_08_21e.dat', '2015_08_21f.dat',
             '2015_08_26f.dat']  # there are probably more: see labbooks
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/rawData'
    v_rest = None
    correct_vrest = True
    protocol_base = 'rampIV' #IV  #Zap20
    protocol = protocol_base
    reg_exp_protocol = re.compile(protocol_base+'(\([0-9]+\))?')
    save_dir = os.path.join(save_dir, protocol)

    for cell in cells:
        hekareader = HekaReader(os.path.join(data_dir, cell))
        group = 'Group1'
        protocol_to_series = hekareader.get_protocol(group)
        n_protocols = sum([1 if reg_exp_protocol.match(p) else 0 for p in protocol_to_series.keys()])
        vms = []
        for i in range(n_protocols):
            if i == 0:
                protocol = protocol_base
            else:
                protocol = protocol_base+'('+str(i)+')'
            # TODO sweep_idx = [0]
            sweep_idx = [-1]
            indices = get_indices(group, sweep_idx)
            if indices is None:
                continue
            else:
                index = indices[0]
            t, vm = hekareader.get_xy(index)
            t *= 1000  # ms
            vm *= 1000  # mV
            if correct_vrest:
                vm = correct_baseline(vm, v_rest)
            vms.append(vm)

        # plot
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if len(vms) >= 2:
            pl.figure()
            pl.title(cell, fontsize=16)
            pl.plot(t, vms[0], 'k', label='before ZD')
            pl.plot(t, vms[-1], 'b', label='after ZD')
            pl.xlabel('Time (ms)', fontsize=16)
            pl.ylabel('Membrane potential (mV)', fontsize=16)
            pl.legend(loc='lower right', fontsize=16)
            pl.savefig(os.path.join(save_dir, cell[:-3]+'png'))
            pl.show()