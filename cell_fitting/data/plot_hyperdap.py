import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.data import shift_v_rest
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_protocols_same_base
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    save_dir = './plots/hyperdap'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'hyperRampTester'
    protocol2 = 'depoRampTester'
    v_rest_shift = -16
    AP_threshold = -30
    repetition = 0
    cells = get_cells_for_protocol(data_dir, protocol)

    cells = ['2013_12_11b', '2013_12_13f', '2013_12_11a', '2013_12_10b', '2013_12_10h', '2013_12_11c',
             '2013_12_11e', '2013_12_10d', '2013_12_10c', '2013_12_13c', '2013_12_11f', '2013_12_13b']

    #cells_try_other_run = ['2013_12_12e']
    #cells_not_so_good = ['2013_12_12d', '2013_12_13e', '2013_12_13d','2013_12_12b']

    for cell in cells:
        if '2012' in cell:
            continue

        protocols = get_protocols_same_base(os.path.join(data_dir, cell+'.dat'), protocol)
        vs = []
        ts = []
        amps = []
        for i, p in enumerate(protocols):
            if -0.05 + i * -0.05 < -0.2:
                break
            v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell+'.dat'), p,
                                                         return_sweep_idxs=True)
            vs.append(shift_v_rest(v_mat[repetition], v_rest_shift))
            ts.append(t_mat[repetition])
            amps.append(-0.05 + i * -0.05)
        vs = vs[::-1]
        ts = ts[::-1]
        amps = amps[::-1]

        protocols = get_protocols_same_base(os.path.join(data_dir, cell+'.dat'), protocol2)
        for i, p in enumerate(protocols):
            if 0.05 + i * 0.05 > 0.2:
                break
            v_mat, t_mat = get_v_and_t_from_heka(os.path.join(data_dir, cell+'.dat'), p)
            vs.append(shift_v_rest(v_mat[repetition], v_rest_shift))
            ts.append(t_mat[repetition])
            amps.append(0.05 + i * 0.05)

        # plot
        save_dir_fig = os.path.join(save_dir, cell)
        if not os.path.exists(save_dir_fig):
            os.makedirs(save_dir_fig)

        c_map = pl.cm.get_cmap('plasma')
        colors = c_map(np.linspace(0, 1, len(vs)))

        print cell

        pl.figure(figsize=(8, 6))
        for i, (v, t) in enumerate(zip(vs, ts)):
            pl.plot(t, v, color=colors[i], label=str(np.round(amps[i], 2)) + ' nA')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane potential (mV)')
        pl.legend(loc='upper right')
        pl.tight_layout()
        pl.xlim(0, t[-1])
        pl.savefig(os.path.join(save_dir_fig, 'v.png'))
        pl.show()

        pl.figure(figsize=(8, 6))
        for i, (v, t) in enumerate(zip(vs, ts)):
            pl.plot(t, v, color=colors[i], label=str(np.round(amps[i], 2)) + ' nA')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane potential (mV)')
        pl.legend()
        pl.xlim(595, 645)
        pl.ylim(-95, -40)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_fig, 'v_zoom.png'))
        pl.show()