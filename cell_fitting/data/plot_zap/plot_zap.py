import os
import matplotlib.pyplot as pl
import numpy as np
import json
import statsmodels.api as sm
from cell_characteristics import to_idx
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_fitting.optimization.evaluation.plot_zap import get_i_inj_zap
from cell_fitting.optimization.fitfuns import impedance
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, shift_v_rest, set_v_rest
from cell_fitting.data import check_cell_has_DAP
from cell_fitting.util import init_nan
pl.style.use('paper')


if __name__ == '__main__':
    #data_dir = '/home/cfischer/Phd/DAP-Project/cell_data/raw_data'
    data_dir = '/media/cfischer/TOSHIBA EXT/Sicherung_2018_05_19/Phd/DAP-Project/cell_data/raw_data'
    v_rest = None
    v_shift = -16
    protocol = 'Zap20'
    animal = 'rat'
    save_dir = os.path.join('../plots', protocol, animal)

    # get cell_ids
    cell_ids = get_cells_for_protocol(data_dir, protocol)
    cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)
    cell_ids = filter(lambda id: check_cell_has_DAP(id), cell_ids)

    # frequencies
    freq0 = 0
    freq1 = 20
    onset_dur = 2000
    offset_dur = 2000

    freqs = lambda x: (freq0-freq1)/(onset_dur-t[-1]-offset_dur) * x \
                      + (freq0-(freq0-freq1)/(onset_dur-t[-1]-offset_dur)*onset_dur)
    heavy_freqs = lambda x: freqs(x) if onset_dur < x < t[-1]-offset_dur else 0
    freqs_out = lambda x: "%.2f" % heavy_freqs(x)

    res_freqs = init_nan(len(cell_ids))
    q_values = init_nan(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        v = shift_v_rest(v_mat[0], v_shift)
        t = t_mat[0]
        dt = t[1]-t[0]
        tstop = t[-1]
        i_inj = get_i_inj_zap(0.1, freq0=freq0, freq1=freq1, onset_dur=onset_dur, offset_dur=offset_dur, dt=dt)

        # cut off onset and offset and downsample
        ds = 1000  # number of steps skipped (in t, i, v) for the impedance computation
        t_ds = t[to_idx(onset_dur, dt, 3):to_idx(tstop-offset_dur, dt, 3):ds]
        i_inj_ds = i_inj[to_idx(onset_dur, dt, 3):to_idx(tstop-offset_dur, dt, 3):ds]
        v_ds = v[to_idx(onset_dur, dt, 3):to_idx(tstop-offset_dur, dt, 3):ds]

        # compute impedance
        imp, frequencies, imp_smooth = impedance(v_ds, i_inj_ds, (t_ds[1] - t_ds[0]) / 1000, [freq0, freq1])  # dt in (sec) for fft

        # resonance frequency and q value
        res_freq_idx = np.argmax(imp_smooth)
        res_freqs[cell_idx] = frequencies[res_freq_idx]
        q_values[cell_idx] = imp_smooth[res_freq_idx] / imp_smooth[np.where(frequencies == 0)[0][0]]

        # plot
        save_dir_img = os.path.join(save_dir, animal, cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        # save
        if not os.path.exists(os.path.join(save_dir, cell_id)):
            os.makedirs(os.path.join(save_dir, cell_id))

        impedance_dict = dict(impedance=list(imp_smooth), frequencies=list(frequencies))
        with open(os.path.join(save_dir, cell_id, 'impedance_dict.json'), 'w') as f:
            json.dump(impedance_dict, f)

        np.save(os.path.join(save_dir_img, 'impedance.npy'), imp_smooth)
        np.save(os.path.join(save_dir_img, 'frequencies.npy'), frequencies)

        # plot impedance
        # pl.figure()
        # pl.plot(frequencies, imp, c='k', label='impedance')
        # pl.plot(frequencies, imp_smooth, label='smoothed impedance', color='r')
        # pl.legend()
        # pl.xlim(1, freq1)
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_img, 'impedance.png'))
        # pl.show()

        # plot v
        # fig, ax1 = pl.subplots()
        # ax2 = ax1.twinx()
        # ax1.plot(t, v, 'k', label='Exp. Data')
        # ax1.set_xlim(0, t[-1])
        # ax2.set_xlim(ax1.get_xlim())
        # ax2.set_xticks(ax1.get_xticks())
        # ax2.set_xticklabels(map(freqs_out, ax1.get_xticks()))
        # ax2.spines['top'].set_visible(True)
        # ax2.set_xlabel('Frequency (Hz)')
        # ax1.set_xlabel('Time (ms)')
        # ax1.set_ylabel('Membrane potential (mV)')
        # # ax1.legend()
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_img, 'v.png'))
        # # pl.show()


        # plot v and impedance
        v_rest = -75
        v = set_v_rest(v, v[0], v_rest)  # use same v_rest

        fig, ax1 = pl.subplots()
        ax2 = ax1.twinx().twiny()  # need two twins for labeling new x and y axis
        ax3 = ax1.twiny().twinx()
        ax1.plot((t - onset_dur)/1000, v, 'k')
        #ylim = ax1.get_ylim()
        #ax1.set_ylim(ylim[0]-2, ylim[1]+2)
        ax1.set_ylim(v_rest-10, v_rest+10)
        ax1.set_xlim(0, (tstop-offset_dur-onset_dur)/1000)
        ax2.plot(frequencies, imp_smooth, c='r', label='Res. Freq.: %.2f (Hz)' % res_freqs[cell_idx] + '\nQ-Value: %.2f' % q_values[cell_idx])
        ax3.plot(frequencies, imp_smooth, c='r')
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax2.set_xlim(freq0, freq1)
        ax3.set_xlim(freq0, freq1)
        #ylim = ax3.get_ylim()
        ylim = [0, 100]
        ax3.set_ylim(ylim[0]-5, ylim[1]+5)
        ax2.set_ylim(ylim[0]-5, ylim[1]+5)
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(True)
        ax2.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Impedance (M$\Omega$)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Membrane Potential (mV)')

        # ax2.annotate('%.2f (Hz)' % res_freq, xy=(res_freq, imp_smooth[res_freq_idx]+0.3),
        #              xytext=(res_freq+0.5, imp_smooth[res_freq_idx]+3), fontsize=16,
        #             arrowprops=dict(arrowstyle='wedge', color='r'))
        leg = ax2.legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=16)
        for item in leg.legendHandles:
            item.set_visible(False)

        pl.tight_layout()
        pl.subplots_adjust(left=0.18, right=0.86, bottom=0.14, top=0.88)
        pl.savefig(os.path.join(save_dir_img, 'v_impedance.png'))
        #pl.show()
        pl.close('all')

    save_dir_summary = os.path.join(save_dir, 'summary')
    if not os.path.exists(save_dir_summary):
        os.makedirs(save_dir_summary)

    np.save(os.path.join(save_dir_summary, 'res_freqs.npy'), res_freqs)
    np.save(os.path.join(save_dir_summary, 'q_values.npy'), q_values)
    np.save(os.path.join(save_dir_summary, 'cell_ids.npy'), cell_ids)