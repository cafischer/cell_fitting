import os
import numpy as np
import matplotlib.pyplot as pl
from cell_fitting.read_heka.i_inj_functions import get_i_inj_zap
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.optimization.fitfuns import impedance
from cell_characteristics import to_idx
from cell_fitting.read_heka import set_v_rest
from cell_fitting.optimization.evaluation import joint_plot_data_and_model


def evaluate_zap(pdf, cell, data_dir_resonance, save_dir):
    save_dir_img = os.path.join(save_dir, 'img', 'zap')
    amp = 0.1
    freq0 = 0
    freq1 = 20
    onset_dur = 2000
    offset_dur = 2000
    zap_dur = 30000
    tstop = 34000
    dt = 0.01

    # simulate / load
    v, t, i_inj = simulate_zap(cell, amp, freq0, freq1, onset_dur, offset_dur, zap_dur, tstop, dt)

    # evaluate
    res_freqs_data = np.load(os.path.join(data_dir_resonance, 'res_freqs.npy'))
    q_values_data = np.load(os.path.join(data_dir_resonance, 'q_values.npy'))

    imp_smooth, frequencies = compute_smoothed_impedance(v, freq0, freq1, i_inj, offset_dur, onset_dur, tstop, dt)
    res_freq, q_value = compute_res_freq_and_q_val(imp_smooth, frequencies)

    v_rest = -75
    v = set_v_rest(v, v[0], v_rest)  # for plotting bring to same v_rest

    # plot in pdf
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    fig = plot_v_and_impedance(freq0, freq1, frequencies, imp_smooth, offset_dur, onset_dur, q_value, res_freq,
                               save_dir_img, t, tstop, v, v_rest)
    pdf.savefig(fig)
    pl.close()

    fig = joint_plot_data_and_model(res_freqs_data, q_values_data, res_freq, q_value, 'Res. Freq.', 'Q-Value',
                                    save_dir_img)
    pdf.savefig(fig)
    pl.close()


def compute_smoothed_impedance(v, freq0, freq1, i_inj, offset_dur, onset_dur, tstop, dt):
    i_inj_ds = i_inj[to_idx(onset_dur, dt, 3):to_idx(tstop - offset_dur, dt, 3)]
    v_ds = v[to_idx(onset_dur, dt, 3):to_idx(tstop - offset_dur, dt, 3)]
    imp, frequencies, imp_smooth = impedance(v_ds, i_inj_ds, dt / 1000, [freq0, freq1])  # dt in (sec) for fft
    return imp_smooth, frequencies

def compute_res_freq_and_q_val(imp_smooth, frequencies):
    res_freq_idx = np.argmax(imp_smooth)
    res_freq = frequencies[res_freq_idx]
    q_value = imp_smooth[res_freq_idx] / imp_smooth[np.where(frequencies == 0)[0][0]]
    return res_freq, q_value


def simulate_zap(cell, amp=0.1, freq0=0, freq1=20, onset_dur=2000, offset_dur=2000, zap_dur=30000,
                 tstop = 34000, dt=0.01, v_init=-75, celsius=35, onset=200):
    i_exp = get_i_inj_zap(amp=amp, freq0=freq0, freq1=freq1, onset_dur=onset_dur, offset_dur=offset_dur,
                          zap_dur=zap_dur, tstop=tstop, dt=dt)
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init, 'tstop': tstop,
                         'dt': dt, 'celsius': celsius, 'onset': onset}
    v, t, i_inj = iclamp_handling_onset(cell, **simulation_params)
    return v, t, i_inj


def plot_v_and_impedance(freq0, freq1, frequencies, imp_smooth, offset_dur, onset_dur, q_value, res_freq, save_dir_img,
                         t, tstop, v, v_rest):
    fig, ax1 = pl.subplots()
    ax2 = ax1.twinx().twiny()  # need two twins for labeling new x and y axis
    ax3 = ax1.twiny().twinx()
    ax1.plot((t - onset_dur) / 1000, v, 'k')
    ax1.set_ylim(v_rest - 10, v_rest + 10)
    ax1.set_xlim(0, (tstop - offset_dur - onset_dur) / 1000)
    ax2.plot(frequencies, imp_smooth, c='r', label='Res. Freq.: %.2f (Hz)' % res_freq + '\nQ-Value: %.2f' % q_value)
    ax3.plot(frequencies, imp_smooth, c='r')
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax2.set_xlim(freq0, freq1)
    ax3.set_xlim(freq0, freq1)
    ylim = [0, 100]
    ax3.set_ylim(ylim[0] - 5, ylim[1] + 5)
    ax2.set_ylim(ylim[0] - 5, ylim[1] + 5)
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    ax2.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Impedance (M$\Omega$)', color='r')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Membrane Potential (mV)')
    leg = ax2.legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=16)
    for item in leg.legendHandles:
        item.set_visible(False)
    pl.tight_layout()
    pl.subplots_adjust(left=0.18, right=0.86, bottom=0.14, top=0.88)
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'v_impedance.png'))
    return fig