import matplotlib.pyplot as pl
import os
import numpy as np
from cell_fitting.optimization.evaluation.plot_rampIV import simulate_rampIV
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.util import get_channel_dict_for_plotting, get_channel_color_for_plotting
from neuron import h
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
pl.style.use('paper')


class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        height = 6
        width = 2
        xdescent = 19
        center = 0.5 * width + 0.5 * xdescent, 0.5 * height + 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width, height=height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def block_channel(cell, channel_name, percent_block):
    if isinstance(channel_name, list):
        for p_b, c_n in zip(percent_block, channel_name):
            if c_n == 'pas':
                old_gbar = cell.get_attr(['soma', '0.5', c_n, 'g'])
                new_gbar = old_gbar * (100 - p_b) / 100.
                cell.update_attr(['soma', '0.5', c_n, 'g'], new_gbar)
            else:
                old_gbar = cell.get_attr(['soma', '0.5', c_n, 'gbar'])
                new_gbar = old_gbar * (100 - p_b) / 100.
                cell.update_attr(['soma', '0.5', c_n, 'gbar'], new_gbar)
    else:
        if channel_name == 'pas':
            old_gbar = cell.get_attr(['soma', '0.5', channel_name, 'g'])
            new_gbar = old_gbar * (100 - percent_block) / 100.
            cell.update_attr(['soma', '0.5', channel_name, 'g'], new_gbar)
        else:
            old_gbar = cell.get_attr(['soma', '0.5', channel_name, 'gbar'])
            new_gbar = old_gbar * (100 - percent_block) / 100.
            cell.update_attr(['soma', '0.5', channel_name, 'gbar'], new_gbar)


def block_channel_at_timepoint(cell, channel_name, percent_block, timepoint):
    event = ChangeConductanceEvent(cell, channel_name, percent_block, timepoint)


def plot_channel_block_on_ax(ax, channel_list, t, v_before_block, v_after_block, percent_block, label=True, color='k'):
    channel_dict = get_channel_dict_for_plotting()
    channel_color = get_channel_color_for_plotting()
    h1, = ax.plot(t, v_before_block, color, label='without block') #, markevery=1000, marker='|')

    p = np.arange(0, len(t), 500)
    x = t[p]
    y = v_before_block[p]
    ells = [mpatches.Ellipse(xy=(x[i], y[i]), width=0.6, height=2.6, angle=0, color='k') for i in range(len(x))]
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
    handles = np.zeros(len(channel_list), dtype=object)
    labels = np.zeros(len(channel_list), dtype=object)
    for i, channel_name in enumerate(channel_list):
        handles[i], = ax.plot(t, v_after_block[i, :], color=channel_color[channel_name],
                label=str(percent_block) + '% block of ' + channel_dict[channel_name])
        labels[i] = str(percent_block) + '% block of ' + channel_dict[channel_name]
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mem. pot. (mV)')
    if label:
        ax.legend([(h1, e)]+handles.tolist(), ['without block']+labels.tolist(), loc='upper right',
                  handler_map={mpatches.Ellipse: HandlerEllipse()})


class ChangeConductanceEvent(object):
    def __init__(self, cell, channel_name, percent_block, t_event):
        self.cell = cell
        self.t_event = t_event
        self.channel_name = channel_name
        self.percent_block = percent_block
        self.fih = h.FInitializeHandler(1, self.start_event)

    def start_event(self):
        h.cvode.event(self.t_event, self.set_gbar)

    def set_gbar(self):
        block_channel(self.cell, self.channel_name, self.percent_block)

        if h.cvode.active():
            h.cvode.re_init()
        else:
            h.fcurrent()


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    model_ids = range(2, 3)
    mechanism_dir = '../../../model/channels/vavoulis'
    ramp_amp = 3.5
    onset = 200
    load_mechanism_dir(mechanism_dir)
    channel_names = ['hcn_slow', 'nat', 'nap', 'kdr']  # ['hcn_slow', 'nat', 'nap', 'kdr']
    percent_blocks = [100]  # [5, 10, 20, 50, 100]

    for model_id in model_ids:
        for channel_name in channel_names:
            for percent_block in percent_blocks:
                # load model
                cell = Cell.from_modeldir(os.path.join(save_dir, str(model_id), 'cell.json'))

                # simulation
                v_before, t_before, _ = simulate_rampIV(cell, ramp_amp, v_init=-75, onset=onset)

                # blocking
                # block_channel(cell, channel_name, percent_block)
                block_channel_at_timepoint(cell, channel_name, percent_block, 15+onset)

                # simulation
                v_after, t_after, _ = simulate_rampIV(cell, ramp_amp, v_init=-75, onset=onset)

                # plot
                save_dir_img = os.path.join(save_dir, str(model_id), 'img', 'blocking', 'rampIV', str(channel_name),
                                            str(percent_block))
                if not os.path.exists(save_dir_img):
                    os.makedirs(save_dir_img)
                print model_id
                pl.figure()
                pl.plot(t_before, v_before, 'r', label='before block')
                pl.plot(t_after, v_after, 'b', label='after block')
                pl.xlabel('Time (ms)')
                pl.ylabel('Membrane potential (mV)')
                pl.legend(loc='upper right')
                pl.tight_layout()
                #pl.savefig(os.path.join(save_dir_img, str(ramp_amp) + '(nA).png'))
                #pl.show()

                pl.figure()
                pl.plot(t_before, v_before, 'r', label='before block')
                pl.plot(t_after, v_after, 'b', label='after block')
                pl.xlabel('Time (ms)')
                pl.ylabel('Membrane potential (mV)')
                pl.legend(loc='upper right')
                pl.xlim(5, 80)
                pl.tight_layout()
                #pl.savefig(os.path.join(save_dir_img, str(ramp_amp) + '(nA)_zoom.png'))
                pl.show()