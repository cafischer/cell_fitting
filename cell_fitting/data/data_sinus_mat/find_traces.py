import os
import matplotlib.pyplot as pl
import numpy as np
from cell_fitting.data.data_sinus_mat.read_sinus_mat import get_sinus_data_from_mat
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
pl.style.use('paper')


def find_sine_trace(amp1_to_find=None, amp2_to_find=None, freq1_to_find=None, freq2_to_find=None):
    save_dir = './sinus_mat_files'
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


if __name__ == '__main__':
    v_mat, t_mat, cell_ids, amp1s, _, _, _ = find_sine_trace(None, 0, None, None)

    pl.figure()
    for v, t in zip(v_mat, t_mat):
        pl.plot(t, v)
    pl.show()
    print amp1s