import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function, get_cells_for_protocol, \
    get_sweep_index_for_amp
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import compute_v_sag_and_steady_state
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    AP_threshold = 0
    v_shift = -16
    animal = 'rat'
    amp = -0.1
    save_dir_img = os.path.join('../plots/IV/sag_hist', animal, str(amp))

    cells = get_cells_for_protocol(data_dir, 'IV')
    #cells = ['2015_05_26d', '2015_06_08a', '2015_06_09f', '2015_06_19i', '2015_08_10g', '2015_08_26b']
    sweep_idx = get_sweep_index_for_amp(amp, 'IV')

    sag_amps = []
    v_deflections = []

    for cell_id in cells:
        if not check_rat_or_gerbil(cell_id) == animal:
            continue

        # read data
        try:
            v_mat_data, t_mat_data = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), 'IV',
                                                                       sweep_idxs=[sweep_idx])
        except IndexError:  # if amplitude was not tested, raises IndexError
            continue
        i_inj_mat = get_i_inj_from_function('IV', [sweep_idx], t_mat_data[0][-1], t_mat_data[0][1]-t_mat_data[0][0])
        start_step_idx = np.nonzero(i_inj_mat[0])[0][0]
        end_step_idx = np.nonzero(i_inj_mat[0])[0][-1] + 1

        # compute sag
        v_sags, v_steady_states, _ = compute_v_sag_and_steady_state(v_mat_data, [amp], AP_threshold,
                                                                                   start_step_idx, end_step_idx)
        try:
            sag_amp = v_steady_states[0] - v_sags[0]
            if sag_amp >= 0:
                sag_amps.append(sag_amp)

                vrest = np.mean(v_mat_data[0, :start_step_idx])
                v_deflection = vrest - v_steady_states[0]
                v_deflections.append(v_deflection)

                # if v_deflection > 30 and sag_amp > 10:
                #     pl.figure()
                #     pl.plot(t_mat_data[0], v_mat_data[0])
                #     pl.show()
            else:
                print 'neg sag: ' + str(cell_id)
        except IndexError:
            print 'Has APs: ' + str(cell_id)
            # pl.figure()
            # pl.plot(t_mat_data[0], v_mat_data[0])
            # pl.show()

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