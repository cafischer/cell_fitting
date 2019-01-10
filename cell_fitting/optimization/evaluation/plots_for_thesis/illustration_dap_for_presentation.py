import numpy as np
import os
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from nrn_wrapper import Cell
from cell_fitting.read_heka import load_data
from cell_fitting.optimization.evaluation import simulate_model, get_spike_characteristics_dict
from cell_fitting.optimization.evaluation.plot_rampIV import simulate_rampIV
from cell_fitting.optimization.errfuns import rms
from cell_fitting.optimization.simulate import get_standard_simulation_params
from cell_fitting.util import characteristics_dict_for_plotting
from cell_characteristics.analyze_APs import get_spike_characteristics
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/defense/intro_DAP'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    exp_cell_dr = '2015_08_06d'
    color_exp = '#0099cc'
    color_model = 'k'
    ramp_amp = 3.5
    standard_sim_params = get_standard_simulation_params()

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'), mechanism_dir)

    # plot
    fig = pl.figure(figsize=(6, 5))
    outer = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.18)

    # Explanation DAP characteristics
    ax = pl.Subplot(fig, outer[0, 0])
    fig.add_subplot(ax)

    v_exp, t_exp, i_inj = load_data(os.path.join(save_dir_data, exp_cell + '.dat'), 'rampIV', ramp_amp)

    characteristics = ['AP_max_idx', 'fAHP_min_idx', 'DAP_max_idx', 'DAP_width_idx']
    start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
    v_rest = np.mean(v_exp[0:start_i_inj])
    characteristics_exp = np.array(get_spike_characteristics(v_exp, t_exp, characteristics, v_rest,
                                                             std_idx_times=(0, 1), check=False,
                                                             **get_spike_characteristics_dict(for_data=True)),
                                   dtype=int)
    t_shift = -8
    t_exp += t_shift
    ax.plot(t_exp, v_exp, 'k')
    ax.set_xlim(0, 145)
    ax.set_ylim(-85, 40)
    ax.set_ylabel('Mem. pot. (mV)')
    ax.get_yaxis().set_label_coords(-0.08, 0.5)

    ax.annotate('mAHP', xy=(63., -79), verticalalignment='top', horizontalalignment='center')
    # ax.annotate('fAHP', xy=(t_exp[characteristics_exp[1]], v_exp[characteristics_exp[1]] - 0.5),
    #                verticalalignment='top', horizontalalignment='center')
    # ax.annotate('DAP', xy=(16 + t_shift, -70))

    # inset
    axins = inset_axes(ax, width='60%', height='60%', loc=1)
    axins.plot(t_exp, v_exp, 'k')
    axins.annotate('fAHP', xy=(t_exp[characteristics_exp[1]], v_exp[characteristics_exp[1]] - 0.5),
                   verticalalignment='top', horizontalalignment='center')
    axins.annotate('DAP', xy=(15.5+t_shift, -69))
    axins.axhline(v_rest, linestyle='--', color='0.5')  # helper line v_rest

    axins.set_ylim(v_rest - 2, -50)
    axins.set_xlim(9.+t_shift, 35+t_shift)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)
    ax.plot(t_exp, i_inj, 'k')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (nA)')
    ax.get_yaxis().set_label_coords(-0.08, 0.5)
    ax.set_xlim(0, 145)

    pl.tight_layout()
    pl.subplots_adjust(left=0.11, right=0.99, top=0.98, bottom=0.1)
    pl.savefig(os.path.join(save_dir_img, 'illustration_dap.png'))
    pl.show()