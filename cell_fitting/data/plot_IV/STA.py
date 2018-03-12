from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
from cell_fitting.data.plot_IV import check_v_at_i_inj_0_is_at_right_sweep_idx
from cell_characteristics.sta_stc import find_APs_in_v_trace, get_sta, plots_sta, get_stc, choose_eigvecs, \
    project_back, plots_stc, group_by_AP_max, plot_group_by_AP_max, plot_ICA, plot_all_in_one, plot_PCA_3D, \
    plot_backtransform, plot_ICA_3D, plot_clustering_kmeans
from cell_characteristics import to_idx
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function
from sklearn.decomposition import FastICA
pl.style.use('paper')


if __name__ == '__main__':
    cell_ids = ['2015_08_26b', '2015_08_26e', '2015_08_26f', '2015_08_06a', '2015_08_06c', '2015_08_06d']
    #cell_ids = ['2015_08_06d']
    save_dir = '../plots/IV/STA/'
    protocol = 'IV'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'

    AP_threshold = -20
    before_AP_sta = 25
    after_AP_sta = 25
    before_AP_stc = 0
    after_AP_stc = 25
    start_step = 250  # ms
    end_step = 750  # ms

    for cell_id in cell_ids:
        print cell_id

        # load data
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        t = t_mat[0, :]
        dt = t[1] - t[0]
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])

        check_v_at_i_inj_0_is_at_right_sweep_idx(v_mat, i_inj_mat)

        start_step_idx = to_idx(start_step, dt)
        end_step_idx = to_idx(end_step, dt)
        before_AP_idx_sta = to_idx(before_AP_sta, dt)
        after_AP_idx_sta = to_idx(after_AP_sta, dt)
        before_AP_idx_stc = to_idx(before_AP_stc, dt)
        after_AP_idx_stc = to_idx(after_AP_stc, dt)

        v_mat = v_mat[:, start_step_idx:end_step_idx]  # only take v during step current
        t = t[start_step_idx:end_step_idx]

        # save and plot
        save_dir_img = os.path.join(save_dir, str(cell_id))
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        # STA
        v_APs = []
        for v in v_mat:
            v_APs.extend(find_APs_in_v_trace(v, AP_threshold, before_AP_idx_sta, after_AP_idx_sta))
        v_APs = np.vstack(v_APs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx_sta + 1) * dt
        sta, sta_std = get_sta(v_APs)

        plots_sta(v_APs, t_AP, sta, sta_std, save_dir_img)

        # STC & Group by AP_max & ICA
        v_APs = []
        for v in v_mat:
            v_APs.extend(find_APs_in_v_trace(v, AP_threshold, before_AP_idx_stc, after_AP_idx_stc))
        v_APs = np.vstack(v_APs)
        v_APs_centered = v_APs - np.mean(v_APs, 0)
        t_AP = np.arange(after_AP_idx_stc + before_AP_idx_stc + 1) * dt

        if len(v_APs) > 10:
            # STC
            eigvals, eigvecs, expl_var = get_stc(v_APs)
            chosen_eigvecs = choose_eigvecs(eigvecs, eigvals, n_eigvecs=3)
            back_projection = project_back(v_APs, chosen_eigvecs)
            plots_stc(v_APs, t_AP, back_projection, chosen_eigvecs, expl_var, save_dir_img)

            # Group by AP_max
            mean_high, std_high, mean_low, std_low, AP_max_high_labels = group_by_AP_max(v_APs)
            plot_group_by_AP_max(mean_high, std_high, mean_low, std_low, t_AP, save_dir_img)
            mean_high_centered = mean_high - np.mean(v_APs, 0)
            mean_low_centered = mean_low - np.mean(v_APs, 0)

            # ICA
            ica = FastICA(n_components=3, whiten=True)
            ica_source = ica.fit_transform(v_APs_centered)
            plot_ICA(v_APs, t_AP, ica.mixing_, save_dir_img)

            # plot together
            plot_all_in_one(v_APs, t_AP, back_projection, mean_high, std_high, mean_low, std_low,
                            chosen_eigvecs, expl_var, ica.mixing_, save_dir_img)
            plot_backtransform(v_APs_centered, t_AP, mean_high_centered, mean_low_centered, std_high, std_low,
                               chosen_eigvecs, ica_source, ica.mixing_, save_dir_img)


            plot_PCA_3D(v_APs_centered, chosen_eigvecs, AP_max_high_labels, save_dir_img)
            plot_ICA_3D(v_APs_centered, ica_source, AP_max_high_labels, save_dir_img)
            pl.close('all')
            plot_clustering_kmeans(v_APs_centered, chosen_eigvecs, 4, save_dir_img)
            pl.show()
        pl.close('all')