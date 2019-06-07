import numpy as np
import os
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb
from nrn_wrapper import Cell
from cell_fitting.util import change_color_brightness
from cell_fitting.test_channels.channel_characteristics import boltzmann_fun, time_constant_curve
from cell_fitting.util import get_channel_dict_for_plotting, get_channel_color_for_plotting
from cell_fitting.test_channels.test_ionchannel import current_subtraction, plot_i_steps_on_ax
from cell_fitting.optimization.simulate import get_standard_simulation_params

pl.style.use('paper')


def plot_act_inact_on_ax(ax, v_range, steadystate_act, steadystate_inact, time_constanct_act, time_constanct_inact,
                         channel_name):
    channel_dict = get_channel_dict_for_plotting()
    channel_color = get_channel_color_for_plotting()

    ax_twin = ax.twinx()
    ax.spines['right'].set_visible(True)

    if steadystate_act is not None:
        ax.plot(v_range, steadystate_act,
                color=change_color_brightness(to_rgb(channel_color[channel_name]), 60, 'brighter'),
                label=channel_dict[channel_name] + ' m')
    if steadystate_inact is not None:
        ax.plot(v_range, steadystate_inact,
                color=change_color_brightness(to_rgb(channel_color[channel_name]), 60, 'darker'),
                label=channel_dict[channel_name] + ' h')
    if time_constanct_act is not None:
        ax_twin.plot(v_range, time_constanct_act, linestyle=':',
                color=change_color_brightness(to_rgb(channel_color[channel_name]), 60, 'brighter'))
    if time_constanct_inact is not None:
        ax_twin.plot(v_range, time_constanct_inact, linestyle=':',
                color=change_color_brightness(to_rgb(channel_color[channel_name]), 60, 'darker'))
    ax.set_xlabel('Mem. pot. (mV)')
    ax.set_ylabel('Degree of opening')
    ax_twin.set_ylabel(r'$\mathrm{\tau}$ (ms)')
    ax.set_ylim(0, 1)
    ax_twin.set_ylim(0, None)
    ax.legend(loc='upper right')


if __name__ == '__main__':
    save_dir_img = '/home/cfischer/Dropbox/thesis/figures_results/new'
    save_dir_model = '/home/cfischer/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cfischer/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    save_dir_data = '/home/cfischer/Phd/DAP-Project/cell_data/raw_data'
    save_dir_data_plots = '/home/cfischer/Phd/programming/projects/cell_fitting/cell_fitting/data/plots'
    model = '2'
    exp_cell = '2015_08_26b'
    ramp_amp = 3.5
    standard_sim_params = get_standard_simulation_params()
    colors = ['y', 'xkcd:orange', 'xkcd:red', 'm', 'b']

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell_rounded.json'), mechanism_dir)

    # plot
    fig = pl.figure(figsize=(8, 9))
    outer = gridspec.GridSpec(4, 2)

    # activation and inactivation

    # NaT
    ax = pl.Subplot(fig, outer[0, 0])
    fig.add_subplot(ax)
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.25, 1.05, 'A', transform=ax.transAxes, size=18, weight='bold')

    # steady-state
    v_range = np.arange(-95, 30, 0.1)
    curve_act = boltzmann_fun(v_range, cell.soma(.5).nat.m_vh, -cell.soma(.5).nat.m_vs)
    curve_inact = boltzmann_fun(v_range, cell.soma(.5).nat.h_vh, -cell.soma(.5).nat.h_vs)
    #ax.plot(v_range, curve_act*curve_inact, 'k') # TODO

    # time constants
    time_constanct_act = time_constant_curve(v_range, cell.soma(.5).nat.m_tau_min, cell.soma(.5).nat.m_tau_max,
                                             cell.soma(.5).nat.m_tau_delta, curve_act,
                                             cell.soma(.5).nat.m_vh, cell.soma(.5).nat.m_vs)
    time_constanct_inact = time_constant_curve(v_range, cell.soma(.5).nat.h_tau_min, cell.soma(.5).nat.h_tau_max,
                                               cell.soma(.5).nat.h_tau_delta, curve_inact,
                                               cell.soma(.5).nat.h_vh, cell.soma(.5).nat.h_vs)

    plot_act_inact_on_ax(ax, v_range, curve_act, curve_inact, time_constanct_act, time_constanct_inact, 'nat')
    ax.set_xlim(-95, 30)
    print 'Nat m min tau: ', np.min(time_constanct_act)
    print 'Nat m max tau: ', np.max(time_constanct_act)
    print 'Nat h min tau: ', np.min(time_constanct_inact)
    print 'Nat h max tau: ', np.max(time_constanct_inact)
    
    # NaP
    ax = pl.Subplot(fig, outer[1, 0])
    fig.add_subplot(ax)
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.25, 1.05, 'B', transform=ax.transAxes, size=18, weight='bold')

    # steady-state
    v_range = np.arange(-95, 30, 0.1)
    curve_act = boltzmann_fun(v_range, cell.soma(.5).nap.m_vh, -cell.soma(.5).nap.m_vs)
    curve_inact = boltzmann_fun(v_range, cell.soma(.5).nap.h_vh, -cell.soma(.5).nap.h_vs)
    #ax.plot(v_range, curve_act * curve_inact, 'k')  # TODO

    # time constants
    time_constanct_act = time_constant_curve(v_range, cell.soma(.5).nap.m_tau_min, cell.soma(.5).nap.m_tau_max,
                                             cell.soma(.5).nap.m_tau_delta, curve_act,
                                             cell.soma(.5).nap.m_vh, cell.soma(.5).nap.m_vs)
    time_constanct_inact = time_constant_curve(v_range, cell.soma(.5).nap.h_tau_min, cell.soma(.5).nap.h_tau_max,
                                               cell.soma(.5).nap.h_tau_delta, curve_inact,
                                               cell.soma(.5).nap.h_vh, cell.soma(.5).nap.h_vs)

    plot_act_inact_on_ax(ax, v_range, curve_act, curve_inact, time_constanct_act, time_constanct_inact, 'nap')
    ax.set_xlim(-95, 30)

    print 'Nap m min tau: ', np.min(time_constanct_act)
    print 'Nap m max tau: ', np.max(time_constanct_act)
    print 'Nap h min tau: ', np.min(time_constanct_inact)
    print 'Nap h max tau: ', np.max(time_constanct_inact)

    # Kdr
    ax = pl.Subplot(fig, outer[2, 0])
    fig.add_subplot(ax)
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.25, 1.05, 'C', transform=ax.transAxes, size=18, weight='bold')

    # steady-state
    v_range = np.arange(-95, 30, 0.1)
    curve_act = boltzmann_fun(v_range, cell.soma(.5).kdr.n_vh, -cell.soma(.5).kdr.n_vs)

    # time constants
    time_constanct_act = time_constant_curve(v_range, cell.soma(.5).kdr.n_tau_min, cell.soma(.5).kdr.n_tau_max,
                                             cell.soma(.5).kdr.n_tau_delta, curve_act,
                                             cell.soma(.5).kdr.n_vh, cell.soma(.5).kdr.n_vs)

    plot_act_inact_on_ax(ax, v_range, curve_act, None, time_constanct_act, None, 'kdr')
    ax.set_xlim(-95, 30)

    print 'Kdr m min tau: ', np.min(time_constanct_act)
    print 'Kdr m max tau: ', np.max(time_constanct_act)
    
    # HCN
    ax = pl.Subplot(fig, outer[3, 0])
    fig.add_subplot(ax)
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    ax.text(-0.25, 1.05, 'D', transform=ax.transAxes, size=18, weight='bold')

    # steady-state
    v_range = np.arange(-95, 30, 0.1)
    curve_inact = boltzmann_fun(v_range, cell.soma(.5).hcn_slow.n_vh, -cell.soma(.5).hcn_slow.n_vs)

    # time constants
    time_constanct_inact = time_constant_curve(v_range, cell.soma(.5).hcn_slow.n_tau_min, cell.soma(.5).hcn_slow.n_tau_max,
                                               cell.soma(.5).hcn_slow.n_tau_delta, curve_inact,
                                               cell.soma(.5).hcn_slow.n_vh, cell.soma(.5).hcn_slow.n_vs)

    plot_act_inact_on_ax(ax, v_range, None, curve_inact, None, time_constanct_inact, 'hcn_slow')
    ax.set_xlim(-95, 30)

    print 'HCN h min tau: ', np.min(time_constanct_inact)
    print 'HCN h max tau: ', np.max(time_constanct_inact)

    # voltage step protocols

    # NaT
    ax = pl.Subplot(fig, outer[0, 1])
    fig.add_subplot(ax)
    ax.get_yaxis().set_label_coords(-0.18, 0.5)
    ax.text(-0.28, 1.05, 'E', transform=ax.transAxes, size=18, weight='bold')

    amps = [-80, -80, -100]
    durs = [10, 50, 0]
    v_steps = np.arange(-60, 30, 20)
    stepamp = 2

    sec_channel = getattr(cell.soma(.5), 'nat')

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, standard_sim_params['celsius'], amps, durs, v_steps,
                                     stepamp, standard_sim_params['pos_i'], standard_sim_params['dt'])
    i_steps = i_steps / np.max(np.max(np.abs(i_steps)))
    plot_i_steps_on_ax(ax, i_steps, v_steps, t, colors)
    ax.set_ylabel('Current (norm.)')
    ax.set_ylim(-1.05, 0.01)
    ax.set_xlim(0, 61)

    # NaP
    ax = pl.Subplot(fig, outer[1, 1])
    fig.add_subplot(ax)
    ax.get_yaxis().set_label_coords(-0.18, 0.5)
    ax.text(-0.28, 1.05, 'F', transform=ax.transAxes, size=18, weight='bold')

    amps = [-80, 0, -80]
    durs = [10, 20, 100]
    v_steps = np.arange(-80, 10, 20)
    stepamp = 3
    sec_channel = getattr(cell.soma(.5), 'nap')

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, standard_sim_params['celsius'], amps, durs, v_steps,
                                     stepamp, standard_sim_params['pos_i'], standard_sim_params['dt'])
    i_steps = i_steps / np.max(np.max(np.abs(i_steps)))
    plot_i_steps_on_ax(ax, i_steps, v_steps, t, colors)
    ax.set_ylabel('Current (norm.)')
    ax.set_ylim(-1.05, 0.01)
    ax.set_xlim(0, 130)

    # Kdr
    ax = pl.Subplot(fig, outer[2, 1])
    fig.add_subplot(ax)
    ax.get_yaxis().set_label_coords(-0.18, 0.5)
    ax.text(-0.28, 1.05, 'G', transform=ax.transAxes, size=18, weight='bold')

    amps = [-110, -50, -100]
    durs = [150, 50, 150]
    v_steps = np.arange(-50, 40, 20)
    stepamp = 3
    sec_channel = getattr(cell.soma(.5), 'kdr')

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, standard_sim_params['celsius'], amps, durs, v_steps,
                                     stepamp, standard_sim_params['pos_i'], standard_sim_params['dt'])
    i_steps = i_steps / np.max(np.max(np.abs(i_steps)))
    plot_i_steps_on_ax(ax, i_steps, v_steps, t, colors)
    ax.set_ylabel('Current (norm.)')
    ax.set_ylim(-0.01, 1.05)
    ax.set_xlim(195, 305)

    # HCN
    ax = pl.Subplot(fig, outer[3, 1])
    fig.add_subplot(ax)
    ax.get_yaxis().set_label_coords(-0.18, 0.5)
    ax.text(-0.28, 1.05, 'H', transform=ax.transAxes, size=18, weight='bold')

    amps = [-60, -80, -60]
    durs = [20, 1500, 0]
    v_steps = np.arange(-120, -30, 20)
    stepamp = 2
    sec_channel = getattr(cell.soma(.5), 'hcn_slow')

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, standard_sim_params['celsius'], amps, durs, v_steps,
                                     stepamp, standard_sim_params['pos_i'], standard_sim_params['dt'])
    i_steps = i_steps / np.max(np.max(np.abs(i_steps)))
    plot_i_steps_on_ax(ax, i_steps, v_steps, t, colors)
    ax.set_ylabel('Current (norm.)')
    ax.set_ylim(-1.05, 0.01)
    ax.set_xlim(0, 1510)

    pl.tight_layout()
    pl.subplots_adjust(bottom=0.06, top=0.97)
    pl.savefig(os.path.join(save_dir_img, 'ion_channel_identity.png'))
    pl.show()