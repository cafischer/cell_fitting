import os
import matplotlib.pyplot as pl
import numpy as np
from scipy.io import loadmat
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
pl.style.use('paper')


def get_sinus_data_from_mat(save_dir):
    x = loadmat(save_dir)

    amp1 = x['slow_ampl'][0]
    amp2 = x['fast_ampl'][0]
    freq1 = x['slow_freq'][0]
    freq2 = x['fast_freq'][0]
    t = x['xdata'] * 1000  # ms
    v = x['ydata']
    t_i_inj = x['xstim']
    i_inj = x['ystim']

    return v, t, i_inj, t_i_inj, amp1, amp2, freq1, freq2


def find_sine_trace(amp1_to_find=None, amp2_to_find=None, freq1_to_find=None, freq2_to_find=None,
                    save_dir='./sinus_mat_files'):
    animal = 'rat'
    cell_ids = [file_name.split('_')[0].replace('-', '_') for file_name in os.listdir(save_dir)]
    cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)

    cell_ids_found = []
    v_traces = []
    t_traces = []
    amp1s_found = []
    amp2s_found = []
    freq1_found = []
    freq2_found = []

    for cell_id in cell_ids:
        vs, ts, _, _, amp1s, amp2s, freq1s, freq2s = get_sinus_data_from_mat(
            os.path.join(save_dir, cell_id.replace('_', '-') + '_Sinus_variables.mat'))
        respect_lidx = ~np.isnan(np.array([amp1_to_find, amp2_to_find, freq1_to_find, freq2_to_find], dtype=float))
        for v, t, amp1, amp2, freq1, freq2 in zip(vs[0], ts[0], amp1s, amp2s, freq1s, freq2s):
            comp_array = np.array([amp1 == amp1_to_find, amp2 == amp2_to_find,
                                   freq1 == freq1_to_find, freq2 == freq2_to_find])
            if np.all(comp_array[respect_lidx]):
                cell_ids_found.append(cell_id)
                v_traces.append(v[:, 0])  # 0s repetition
                t_traces.append(t[:, 0])
                amp1s_found.append(amp1)
                amp2s_found.append(amp2)
                freq1_found.append(freq1)
                freq2_found.append(freq2)

    return np.array(v_traces), np.array(t_traces), cell_ids_found, amp1s_found, amp2s_found, freq1_found, freq2_found


def find_sine_trace_of_cell(cell_id, amp1_to_find=None, amp2_to_find=None, freq1_to_find=None, freq2_to_find=None,
                            repetition=None, save_dir='./sinus_mat_files'):

    v_traces = []
    t_traces = []
    amp1s_found = []
    amp2s_found = []
    freq1_found = []
    freq2_found = []

    vs, ts, _, _, amp1s, amp2s, freq1s, freq2s = get_sinus_data_from_mat(
        os.path.join(save_dir, cell_id.replace('_', '-') + '_Sinus_variables.mat'))
    respect_lidx = ~np.isnan(np.array([amp1_to_find, amp2_to_find, freq1_to_find, freq2_to_find], dtype=float))
    for v, t, amp1, amp2, freq1, freq2 in zip(vs[0], ts[0], amp1s, amp2s, freq1s, freq2s):
        comp_array = np.array([amp1 == amp1_to_find, amp2 == amp2_to_find,
                               freq1 == freq1_to_find, freq2 == freq2_to_find])
        if np.all(comp_array[respect_lidx]):
            if repetition is None:
                for rep in range(np.shape(v)[1]):
                    v_traces.append(v[:, rep])
                    t_traces.append(t[:, rep])
                    amp1s_found.append(amp1)
                    amp2s_found.append(amp2)
                    freq1_found.append(freq1)
                    freq2_found.append(freq2)
            else:
                v_traces.append(v[:, repetition])
                t_traces.append(t[:, repetition])
                amp1s_found.append(amp1)
                amp2s_found.append(amp2)
                freq1_found.append(freq1)
                freq2_found.append(freq2)

    return np.array(v_traces), np.array(t_traces), amp1s_found, amp2s_found, freq1_found, freq2_found


if __name__ == '__main__':
    # example read_data_sinus_mat
    save_dir = './sinus_mat_files'
    cell_id = '2015-08-25d'
    file_name = cell_id + '_Sinus_variables.mat'
    repetition = 0

    # save_dir = os.path.join(save_dir, file_name)
    #
    # v, t, i_inj, t_i_inj, amp1, amp2, freq1, freq2 = get_sinus_data_from_mat(save_dir)
    #
    # print amp2
    # for i in range(len(amp1)):
    #
    #     pl.figure()
    #     pl.title('amp1: ' + str(amp1[i]) + ' ' + 'amp2: ' + str(amp2[i]) + ' ' +
    #              'freq1: ' + str(freq1[i]) + ' ' + 'freq2: ' + str(freq2[i]) + ' ')
    #     pl.plot(t[0, i][:, repetition], v[0, i][:, repetition], 'k', linewidth=1.0)
    #     pl.show()
    #
    #     # pl.figure()
    #     # # pl.title('amp1: '+str(amp1[i])+' '+'amp2: '+str(amp2[i])+' '+
    #     # #          'freq1: '+str(freq1[i])+' '+'freq2: '+str(freq2[i])+' ')
    #     # pl.plot(t_i_inj[0, i][0, :], i_inj[0, i][0, :], 'k')
    #     # pl.xlabel('Time (s)')
    #     # pl.ylabel('Current (nA)')
    #     # pl.tight_layout()
    #     # pl.show()

    # example find_traces
    v_mat, t_mat, amp1s, amp2s, _, _ = find_sine_trace_of_cell(cell_id, None, None, 0.25, 5, save_dir)

    print amp1s[0], amp2s[0]
    pl.figure()
    #for v, t in zip(v_mat, t_mat):
    #    pl.plot(t, v)
    pl.plot(t_mat[0], v_mat[0])
    pl.ylim(-100, 40)
    pl.show()