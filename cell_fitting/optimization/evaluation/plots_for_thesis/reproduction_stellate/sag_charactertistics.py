import numpy as np
import os
import matplotlib.pyplot as pl
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import compute_v_sag_and_steady_state
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_characteristics import to_idx
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_methods'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    exp_cell = '2015_08_26b'
    step_amp = -0.1
    standard_sim_params = get_standard_simulation_params()

    # load data
    v_data, t_data, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'IV', step_amp)
    dt = t_data[1] - t_data[0]

    # compute v_rest
    start_step_idx = np.nonzero(i_inj)[0][0]
    end_step_idx = np.nonzero(i_inj)[0][-1] + 1
    vrest = np.mean(v_data[:start_step_idx])

    # compute sag deflection and amp. at steady-state in the model
    v_sags, v_steady_states, _ = compute_v_sag_and_steady_state([v_data], [step_amp], AP_threshold=0,
                                                                start_step_idx=start_step_idx,
                                                                end_step_idx=end_step_idx)
    sag_amp_model = v_steady_states[0] - v_sags[0]
    v_deflection_model = vrest - v_steady_states[0]
    sag_idx = np.argmin(v_data)

    # plot
    fig, ax = pl.subplots()
    ax.plot(t_data, v_data, 'k')

    idx_steady_arrow = end_step_idx - to_idx(50, dt)
    ax.plot(t_data[start_step_idx:end_step_idx],
            np.ones(len(t_data[start_step_idx:end_step_idx])) * vrest,
            '--', color='0.5')
    ax.annotate('', xy=(t_data[idx_steady_arrow], vrest),
                xytext=(t_data[idx_steady_arrow], v_steady_states[0]),
                arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0})
    ax.annotate('Steady state amp.', xy=(t_data[idx_steady_arrow]-10, vrest+(v_steady_states[0] - vrest)/2.),
                   verticalalignment='center', horizontalalignment='right')

    idx_sag_arrow = start_step_idx - to_idx(20, dt)
    ax.plot(t_data[idx_sag_arrow:to_idx(t_data[sag_idx], dt)],
            np.ones(len(t_data[idx_sag_arrow:to_idx(t_data[sag_idx], dt)])) * v_sags[0],
            '--', color='0.5')
    ax.plot(t_data[idx_sag_arrow:end_step_idx],
            np.ones(len(t_data[idx_sag_arrow:end_step_idx])) * v_steady_states[0],
            '--', color='0.5')
    ax.annotate('', xy=(t_data[idx_sag_arrow], v_sags[0]),
                xytext=(t_data[idx_sag_arrow], v_steady_states[0]),
                arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0})
    ax.annotate('Sag deflection', xy=(t_data[idx_sag_arrow]-10, v_sags[0]+(v_steady_states[0] - v_sags[0])/2.),
                   verticalalignment='center', horizontalalignment='right')

    ax.set_ylabel('Mem. pot. (mV)')
    ax.set_xlabel('Time (ms)')

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'sag_characteristics.png'))
    pl.show()