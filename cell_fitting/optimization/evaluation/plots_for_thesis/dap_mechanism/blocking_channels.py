import numpy as np
import os
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation import simulate_model, simulate_model_currents
from cell_fitting.optimization.evaluation.plot_blocking.block_channel import block_channel, \
    block_channel_at_timepoint, plot_channel_block_on_ax
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_characteristics.analyze_APs import get_spike_characteristics
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    ramp_amp = 3.5
    standard_sim_params = get_standard_simulation_params()
    standard_sim_params['tstop'] = 162

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'), mechanism_dir)

    # simulate cell
    v_model, t_model, i_inj = simulate_model(cell, 'rampIV', ramp_amp, **standard_sim_params)
    currents, channel_list = simulate_model_currents(cell, 'rampIV', ramp_amp, **standard_sim_params)

    # plot
    fig = pl.figure(figsize=(11, 7))
    outer = gridspec.GridSpec(2, 3)

    # blocking ion channels whole trace
    axes = [outer[0, 0], outer[0, 1], outer[0, 2]]
    percent_blocks = [10, 50, 100]
    letters = ['A', 'B', 'C']

    for percent_block_idx, percent_block in enumerate(percent_blocks):
        ax = pl.Subplot(fig, axes[percent_block_idx])
        fig.add_subplot(ax)

        v_after_block = np.zeros((len(channel_list), len(t_model)))
        for i, channel_name in enumerate(channel_list):
            cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))
            block_channel(cell, channel_name, percent_block)
            v_after_block[i, :], _, _ = simulate_model(cell, 'rampIV', ramp_amp, **standard_sim_params)

        plot_channel_block_on_ax(ax, channel_list, t_model, v_model, v_after_block, percent_block)
        ax.set_ylim(-100, 60)
        ax.get_yaxis().set_label_coords(-0.15, 0.5)
        ax.text(-0.25, 1.0, letters[percent_block_idx], transform=ax.transAxes, size=18, weight='bold')

    # blocking ion channels after AP
    axes = [outer[1, 0], outer[1, 1], outer[1, 2]]
    letters = ['D', 'E', 'F']

    start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
    v_rest = np.mean(v_model[0:start_i_inj])
    fAHP_min_idx = get_spike_characteristics(v_model, t_model, ['fAHP_min_idx'], v_rest,
                                             check=False, **get_spike_characteristics_dict())[0]

    for percent_block_idx, percent_block in enumerate(percent_blocks):
        ax = pl.Subplot(fig, axes[percent_block_idx])
        fig.add_subplot(ax)

        v_after_block = np.zeros((len(channel_list), len(t_model)))
        for i, channel_name in enumerate(channel_list):
            cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))
            block_channel_at_timepoint(cell, channel_name, percent_block,
                                       t_model[fAHP_min_idx]+standard_sim_params['onset'])
            v_after_block[i, :], _, _ = simulate_model(cell, 'rampIV', ramp_amp, **standard_sim_params)

        plot_channel_block_on_ax(ax, channel_list, t_model, v_model, v_after_block, percent_block)
        ax.set_ylim(-100, 60)
        ax.get_yaxis().set_label_coords(-0.15, 0.5)
        ax.text(-0.25, 1.0, letters[percent_block_idx], transform=ax.transAxes, size=18, weight='bold')

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'block_channels.png'))
    pl.show()