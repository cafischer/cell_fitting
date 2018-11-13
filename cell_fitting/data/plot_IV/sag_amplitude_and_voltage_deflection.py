import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function, get_cells_for_protocol, \
    get_sweep_index_for_amp, shift_v_rest
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import compute_v_sag_and_steady_state
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_fitting.data import check_cell_has_DAP
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    AP_threshold = 0
    v_shift = -16
    animal = 'rat'
    protocol = 'IV'
    amp = -0.1
    save_dir_img = os.path.join('../plots', protocol, 'sag', animal, str(amp))

    # get cell_ids
    cell_ids = get_cells_for_protocol(data_dir, protocol)
    cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)
    cell_ids = filter(lambda id: check_cell_has_DAP(id), cell_ids)

    sweep_idx = get_sweep_index_for_amp(amp, 'IV')

    sag_amps = []
    v_deflections = []
    for cell_id in cell_ids:
        # read data
        try:
            v_mat, t_mat = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), 'IV',
                                                 sweep_idxs=[sweep_idx])
            v = shift_v_rest(v_mat[0], v_shift)
            t = t_mat[0]
            i_inj_mat = get_i_inj_from_function('IV', [sweep_idx], t[-1], t[1] - t[0])
            i_inj = i_inj_mat[0]
        except IndexError:  # if amplitude was not tested, raises IndexError
            continue
        start_step_idx = np.nonzero(i_inj)[0][0]
        end_step_idx = np.nonzero(i_inj)[0][-1] + 1

        # compute sag
        v_sags, v_steady_states, _ = compute_v_sag_and_steady_state([v], [amp], AP_threshold,
                                                                    start_step_idx, end_step_idx)
        try:
            sag_amp = v_steady_states[0] - v_sags[0]
            if sag_amp >= 0:
                sag_amps.append(sag_amp)

                vrest = np.mean(v[:start_step_idx])
                v_deflection = vrest - v_steady_states[0]
                v_deflections.append(v_deflection)

                # if v_deflection > 30 and sag_amp > 10:
                #     pl.figure()
                #     pl.plot(t, v)
                #     pl.show()
            else:
                print 'Negative sag amp.: ' + str(cell_id)
                pl.figure()
                pl.plot(t, v)
                pl.show()
        except IndexError:
            print 'Has APs: ' + str(cell_id)
            pl.figure()
            pl.plot(t, v)
            pl.show()

    # plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'sag_amps.npy'), sag_amps)
    np.save(os.path.join(save_dir_img, 'v_deflections.npy'), v_deflections)

    pl.figure()
    pl.hist(sag_amps, bins=100, color='0.5')
    pl.xlabel('Sag Amplitude (mV)')
    pl.ylabel('Count')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'sag_hist.png'))

    pl.figure()
    pl.hist(v_deflections, bins=100, color='0.5')
    pl.xlabel('Voltage Deflection (mV)')
    pl.ylabel('Count')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'deflection_hist.png'))
    #pl.show()