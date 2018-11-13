from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
import pandas as pd
from cell_fitting.read_heka import get_v_and_t_from_heka, get_protocols_same_base, shift_v_rest
pl.style.use('paper')


def plot_double_ramp():
    pl.figure()
    for ramp3time_idx in range(len_ramp3_times):
        pl.plot(t, v_mat[ramp3_amp_idx, ramp3time_idx, :, step_idx_dict[step_amp]], 'k',
                label='Exp. Data' if ramp3time_idx == 0 else '')
    pl.ylabel('Membrane Potential (mV)')
    pl.xlabel('Time (ms)')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_cell, 'PP' + str(ramp3_amp_idx) + '.png'))

    pl.figure()
    for ramp3time_idx in range(len_ramp3_times):
        pl.plot(t, v_mat[ramp3_amp_idx, ramp3time_idx, :], 'k', label='Exp. Data' if ramp3time_idx == 0 else '')
    pl.ylabel('Membrane Potential (mV)')
    pl.xlabel('Time (ms)')
    if len_t == 46900:
        pl.xlim(265, 300)
    elif len_t == 50200 or len_t == 50700 or len_t == 56700 or len_t == 57200:
        # pl.xlim(360, 400)
        pl.xlim(295, 330)
    elif len_t == 56400:
        pl.xlim(355, 390)
    elif len_t == 69200:
        pl.xlim(485, 560)
    elif len_t == 93200 or len_t == 93400:
        pl.xlim(725, 760)
    else:
        pl.xlim(485, 560)
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_cell, 'PP' + str(ramp3_amp_idx) + '_zoom.png'))
    # pl.show()
    pl.close()


if __name__ == '__main__':
    save_dir = '../plots'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    PP_params_dir = '/home/cf/Phd/DAP-Project/cell_data/PP_params2.csv'
    protocol = 'PP'
    v_rest_shift = -16
    cell_id = '2014_07_10d'
    run_idx = 5
    step_idx_dict = {-0.1: 0, 0: 1, 0.1: 2}
    step_amps = [0, -0.1, 0.1]

    # read params
    PP_params = pd.read_csv(PP_params_dir, header=0)
    PP_params['cell_id'].fillna(method='ffill', inplace=True)

    file_dir = os.path.join(data_dir, cell_id + '.dat')
    params = PP_params[PP_params['cell_id'] == cell_id].iloc[run_idx]
    start_series = int(params['start_series'])
    end_series = int(params['end_series']) if not np.isnan(params['end_series']) else np.nan
    len_ramp3_times = int(params['len_ramp3_times'])

    # read in all PPs
    protocols, heka_dict = get_protocols_same_base(file_dir, protocol, return_heka=True)
    v_traces = []
    t_traces = []
    series_traces = []
    for protocol in protocols:
        v_tmp, t_tmp, series_tmp = get_v_and_t_from_heka(file_dir, protocol, return_series=True, heka_dict=heka_dict)
        v_traces.append(v_tmp[0, :])
        t_traces.append(t_tmp[0, :])
        series_traces.append(series_tmp)

    # select traces
    start_idx = np.where(np.array(series_traces) == 'Series' + str(start_series))[0][0]
    end_idx = np.where(np.array(series_traces) == 'Series' + str(end_series))[0]
    if len(end_idx) != 1:
        print 'end not found'
    end_idx = end_idx[0] if len(end_idx) == 1 else len(series_traces)-1
    # for i, idx in enumerate(range(start_series, end_series+1)):
    #     if not 'Series'+str(idx) == series_traces[start_idx:end_idx+1][i]:
    #         print 'Not continuous numbering at: ' + 'Series'+str(idx)
    #         break
    v_traces = v_traces[start_idx:end_idx+1]
    t_traces = t_traces[start_idx:end_idx+1]
    len_t = np.min(np.array([len(t) for t in t_traces]))
    print len_t
    len_ramp3_amps = max(1, int(np.floor(len(v_traces) / (len(step_amps) * len_ramp3_times))))
    v_mats = []

    # create v_mat (with dimensions: ramp3_amps, ramp3_times, t, step_amps)
    v_mat = np.zeros((len_ramp3_amps, len_ramp3_times, len_t, len(step_amps)))
    step_idx_dict_data = {-0.1: 1, 0: 0, 0.1: 2}
    for step_amp in step_amps:
        step_str = 'step_%.1f(nA)' % step_amp

        save_dir_cell = os.path.join(save_dir, 'PP', cell_id, str(run_idx), step_str)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        for ramp3_amp_idx in range(len_ramp3_amps):
            for ramp3time_idx in range(len_ramp3_times):
                idx_traces = (ramp3time_idx*len(step_amps)) \
                             + (ramp3_amp_idx*len(step_amps)*len_ramp3_times) + step_idx_dict_data[step_amp]
                if idx_traces < len(v_traces):
                    t = np.array(t_traces[idx_traces])[:len_t]
                    v = np.array(v_traces[idx_traces])[:len_t]
                    v = shift_v_rest(v, v_rest_shift)
                    v_mat[ramp3_amp_idx, ramp3time_idx, :, step_idx_dict[step_amp]] = v

            plot_double_ramp()

    np.save(os.path.join(save_dir, 'PP', cell_id, str(run_idx), 'v_mat.npy'), v_mat)
    np.save(os.path.join(save_dir, 'PP', cell_id, str(run_idx), 't.npy'), t)

# Series number = 2nd number in Igor (sweep number is 3rd) = current number Franzi writes down