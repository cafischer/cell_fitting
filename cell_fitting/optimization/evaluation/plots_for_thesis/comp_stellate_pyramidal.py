import numpy as np
import os
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from cell_fitting.read_heka import load_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_introduction'
    save_dir_data = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    exp_cell_stellate = '2015_08_26b'
    exp_cell_pyramidal = '2015_08_20c'

    fig = pl.figure(figsize=(11, 6.5))
    outer = gridspec.GridSpec(1, 4)

    # DAP
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 0], hspace=0.35, height_ratios=[5, 5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    ax2 = pl.Subplot(fig, inner[2])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)

    v_stellate, t_stellate, i_inj_stellate = load_data(os.path.join(save_dir_data, exp_cell_stellate+'.dat'),
                                                       'rampIV', 3.5)
    v_pyramidal, t_pyramidal, i_inj_pyramidal = load_data(os.path.join(save_dir_data, exp_cell_pyramidal+'.dat'),
                                                          'rampIV', 7.0)

    ax0.plot(t_stellate, v_stellate, 'k', label='Put. stellate')
    ax1.plot(t_pyramidal, v_pyramidal, 'k', label='Put. pyramidal')
    ax2.plot(t_pyramidal, i_inj_pyramidal, 'k', alpha=0.5)
    ax2.plot(t_stellate, i_inj_stellate, 'k')

    ax0.set_xticks([])
    ax1.set_xticks([])
    ax0.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylabel('Mem. pot. (mV)')
    ax2.set_ylabel('Current (nA)')
    ax2.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.get_yaxis().set_label_coords(-0.25, 0.5)
    ax2.get_yaxis().set_label_coords(-0.25, 0.5)
    ax2.set_yticks([np.min(i_inj_pyramidal), np.max(i_inj_pyramidal)])
    ax0.set_xlim(0, t_stellate[-1])
    ax1.set_xlim(0, t_pyramidal[-1])
    ax2.set_xlim(0, t_stellate[-1])

    ax0.text(-0.25, 1.0, 'A', transform=ax0.transAxes, size=18, weight='bold')
    ax1.text(-0.25, 1.0, 'E', transform=ax1.transAxes, size=18, weight='bold')

    # sag
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 1], hspace=0.35, height_ratios=[5, 5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    ax2 = pl.Subplot(fig, inner[2])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)

    step_amp = -0.1
    v_stellate, t_stellate, i_inj = load_data(os.path.join(save_dir_data, exp_cell_stellate + '.dat'), 'IV', step_amp)
    v_pyramidal, t_pyramidal, i_inj = load_data(os.path.join(save_dir_data, exp_cell_pyramidal + '.dat'), 'IV', step_amp)

    ax0.plot(t_stellate, v_stellate, 'k', label='Put. stellate')
    ax1.plot(t_pyramidal, v_pyramidal, 'k', label='Put. pyramidal')
    ax2.plot(t_stellate, i_inj, 'k')

    ax0.set_xticks([])
    ax1.set_xticks([])
    ax2.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.get_yaxis().set_label_coords(-0.25, 0.5)
    ax2.get_yaxis().set_label_coords(-0.25, 0.5)
    ax2.set_yticks([np.min(i_inj), np.max(i_inj)])
    ax0.set_xlim(0, 1000)
    ax1.set_xlim(0, 1000)
    ax2.set_xlim(0, 1000)

    ax0.text(-0.25, 1.0, 'B', transform=ax0.transAxes, size=18, weight='bold')
    ax1.text(-0.25, 1.0, 'F', transform=ax1.transAxes, size=18, weight='bold')

    # pos. step
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 2], hspace=0.35, height_ratios=[5, 5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    ax2 = pl.Subplot(fig, inner[2])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)

    step_amp = 0.35
    v_stellate, t_stellate, i_inj = load_data(os.path.join(save_dir_data, exp_cell_stellate + '.dat'), 'IV', step_amp)
    v_pyramidal, t_pyramidal, i_inj = load_data(os.path.join(save_dir_data, exp_cell_pyramidal + '.dat'), 'IV', step_amp)

    ax0.plot(t_stellate, v_stellate, 'k', label='Put. stellate')
    ax1.plot(t_pyramidal, v_pyramidal, 'k', label='Put. pyramidal')
    ax2.plot(t_stellate, i_inj, 'k')

    axins = inset_axes(ax0, width='20%', height='60%', loc=1)
    axins.plot(t_stellate, v_stellate, 'k')
    axins.set_xlim(250, 290)
    axins.set_ylim(-78, 39)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    mark_inset(ax0, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax0.set_xlim(0, 1000)
    ax1.set_xlim(0, 1000)
    ax2.set_xlim(0, 1000)
    ax0.set_xticks([])
    ax1.set_xticks([])
    ax2.set_xlabel('Time (ms)')
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.get_yaxis().set_label_coords(-0.25, 0.5)
    ax2.get_yaxis().set_label_coords(-0.25, 0.5)
    ax2.set_yticks([np.min(i_inj), np.max(i_inj)])

    ax0.text(-0.25, 1.0, 'C', transform=ax0.transAxes, size=18, weight='bold')
    ax1.text(-0.25, 1.0, 'G', transform=ax1.transAxes, size=18, weight='bold')

    # zap
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 3], hspace=0.35, height_ratios=[5, 5, 1])
    ax0 = pl.Subplot(fig, inner[0])
    ax1 = pl.Subplot(fig, inner[1])
    ax2 = pl.Subplot(fig, inner[2])
    fig.add_subplot(ax0)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)

    step_amp = 0.1
    v_stellate, t_stellate, i_inj = load_data(os.path.join(save_dir_data, exp_cell_stellate + '.dat'), 'Zap20', step_amp)
    v_pyramidal, t_pyramidal, i_inj = load_data(os.path.join(save_dir_data, exp_cell_pyramidal + '.dat'), 'Zap20', step_amp)

    ax0.plot(t_stellate/1000., v_stellate, 'k', label='Put. stellate')
    ax1.plot(t_pyramidal/1000., v_pyramidal, 'k', label='Put. pyramidal')
    ax2.plot(t_stellate/1000., i_inj, 'k')

    ax0.set_xlim(0, t_stellate[-1]/1000.)
    ax1.set_xlim(0, t_pyramidal[-1]/1000.)
    ax2.set_xlim(0, t_stellate[-1]/1000.)
    ax0.set_xticks([])
    ax1.set_xticks([])
    ax2.set_xlabel('Time (s)')
    ax0.get_yaxis().set_label_coords(-0.25, 0.5)
    ax1.get_yaxis().set_label_coords(-0.25, 0.5)
    ax2.get_yaxis().set_label_coords(-0.25, 0.5)
    ax2.set_yticks([np.min(i_inj), np.max(i_inj)])

    ax0.text(-0.25, 1.0, 'D', transform=ax0.transAxes, size=18, weight='bold')
    ax1.text(-0.25, 1.0, 'H', transform=ax1.transAxes, size=18, weight='bold')

    # title
    ax0.annotate('Put. stellate cell', xy=(0.54, 0.96), xycoords='figure fraction', fontsize=14,
                 horizontalalignment='center')
    ax1.annotate('Put. pyramidal cell', xy=(0.54, 0.57), xycoords='figure fraction', fontsize=14,
                 horizontalalignment='center')

    pl.tight_layout()
    pl.subplots_adjust(top=0.92, bottom=0.08)
    pl.savefig(os.path.join(save_dir_img, 'comp_stellate_pyramidal.png'))
    pl.show()