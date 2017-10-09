from scipy.io import loadmat
import os
import matplotlib.pyplot as pl
pl.style.use('paper')


def get_sinus_data_from_mat(save_dir):
    x = loadmat(save_dir)

    amp1 = x['slow_ampl'][0]
    amp2 = x['fast_ampl'][0]
    freq1 = x['slow_freq'][0]
    freq2 = x['fast_freq'][0]
    t = x['xdata']
    v = x['ydata']
    t_i_inj = x['xstim']
    i_inj = x['ystim']

    return v, t, i_inj, t_i_inj, amp1, amp2, freq1, freq2


if __name__ == '__main__':
    save_dir = './sinus_mat_files'
    cell = '2015-05-22r'
    file_name = cell + '_Sinus_variables.mat'
    repetition = 0

    save_dir = os.path.join(save_dir, file_name)
    v, t, i_inj, t_i_inj, amp1, amp2, freq1, freq2 = get_sinus_data_from_mat(save_dir)

    for i in range(len(amp1)):

        pl.figure()
        pl.plot(t[0, i][:, repetition], v[0, i][:, repetition])
        pl.show()

        pl.figure()
        pl.title('amp1: '+str(amp1[i])+' '+'amp2: '+str(amp2[i])+' '+
                 'freq1: '+str(freq1[i])+' '+'freq2: '+str(freq2[i])+' ')
        pl.plot(t_i_inj[0, i][0, :], i_inj[0, i][0, :])
        pl.show()