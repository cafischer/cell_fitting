from __future__ import division
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_fitting.data import shift_v_rest
import os
import matplotlib.pyplot as pl
import re
import numpy as np
import pandas as pd
import json
pl.style.use('paper')


if __name__ == '__main__':
    save_dir = '../plots/sine_stimulus/traces/rat'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    sine_params_dir = '/home/cf/Phd/DAP-Project/cell_data/sine_params.csv'
    protocol_base = 'Stimulus'
    dur1 = 1000  # ms
    freq2 = 5  # Hz
    v_rest_shift = -16
    correct_vrest = True
    dt = 0.05
    reg_exp_protocol = re.compile(protocol_base+'\([0-9]+\)')

    sine_params = pd.read_csv(sine_params_dir)
    sine_params['cell'].ffill(inplace=True)
    sine_params['onset_dur'].ffill(inplace=True)
    sine_params['offset_dur'].ffill(inplace=True)
    sine_params['dt'].ffill(inplace=True)

    animal = 'rat'
    #cell_ids = get_cells_for_protocol(data_dir, protocol_base)
    #cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)


    cell_ids = ['2015_08_10g.dat', '2015_08_25b.dat', '2015_05_29h.dat', '2015_05_29f.dat', '2015_08_26b.dat',
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

    for cell_id in cell_ids:
        sine_params_cell = sine_params[sine_params['cell'] == cell_id[:-4]]
        protocol_idx = np.where(np.logical_and(sine_params_cell['sine1_dur'] == dur1, sine_params_cell['freq2'] == freq2))[0]
        if len(protocol_idx) > 0:
            protocol = protocol_base+'('+str(protocol_idx[0])+')' if protocol_idx[0] > 0 else protocol_base
            v_mat, t_mat = get_v_and_t_from_heka(os.path.join(data_dir, cell_id), protocol)
            t = np.array(t_mat[0])
            v = np.array(v_mat[0])
            v = shift_v_rest(v, v_rest_shift)
            sine_params_cell = sine_params_cell.iloc[protocol_idx[0]].to_dict()

            save_dir_cell = os.path.join(save_dir, str(dur1) + '_' + str(freq2), cell_id[:-4])
            if not os.path.exists(save_dir_cell):
                os.makedirs(save_dir_cell)
            np.save(os.path.join(save_dir_cell, 'v.npy'), v)
            np.save(os.path.join(save_dir_cell, 't.npy'), t)
            with open(os.path.join(save_dir_cell, 'sine_params.json'), 'w') as f:
                json.dump(sine_params_cell, f)

            pl.figure()
            #pl.title(cell)
            pl.plot(t, v, 'k', linewidth=0.7, label='Exp. Data')
            pl.ylabel('Membrane Potential (mV)')
            pl.xlabel('Time (ms)')
            #pl.legend(loc='upper right')
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_cell, 'v.png'))
            pl.show()
