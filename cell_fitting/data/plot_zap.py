import os
import matplotlib.pyplot as pl
from cell_fitting.data import shift_v_rest
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
pl.style.use('paper')


if __name__ == '__main__':

    save_dir = 'plots/'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    v_rest = None
    v_shift = -16
    protocol = 'Zap20'
    animal = 'rat'
    cells = get_cells_for_protocol(data_dir, protocol)
    save_dir = os.path.join(save_dir, protocol, animal)

    # frequencies
    freq0 = 0
    freq1 = 20
    onset_dur = 2000
    offset_dur = 2000

    freqs = lambda x: (freq0-freq1)/(onset_dur-t[-1]-offset_dur) * x \
                      + (freq0-(freq0-freq1)/(onset_dur-t[-1]-offset_dur)*onset_dur)
    heavy_freqs = lambda x: freqs(x) if onset_dur < x < t[-1]-offset_dur else 0
    freqs_out = lambda x: "%.2f" % heavy_freqs(x)

    for cell_id in cells:
        if not check_rat_or_gerbil(cell_id) == animal:
            continue

        v_mat, t_mat = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol)
        v = shift_v_rest(v_mat[0], v_shift)
        t = t_mat[0]

        # plot
        save_dir_img = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        fig, ax1 = pl.subplots()
        ax2 = ax1.twiny()
        ax1.plot(t, v, 'k', label='Exp. Data')
        ax1.set_xlim(0, t[-1])
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels(map(freqs_out, ax1.get_xticks()))
        ax2.spines['top'].set_visible(True)
        ax2.set_xlabel('Frequency (Hz)')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Membrane potential (mV)')
        #ax1.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'v.png'))
        #pl.show()