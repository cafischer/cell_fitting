import os
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as pl
pl.style.use('paper')


# save dir
save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'analysis_2017-10-10_new')
save_dir_plots = os.path.join(save_dir_analysis, 'plots', 'distributions')

if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)

return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))
characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))
candidate_mat = np.load(os.path.join(save_dir_analysis, 'candidate_mat.npy'))

for i, characteristic in enumerate(return_characteristics):
    min_val = np.nanmin(characteristics_mat[:, i])
    max_val = np.nanmax(characteristics_mat[:, i])
    bins = np.linspace(min_val, max_val, 100)
    if return_characteristics[i] == 'AP_width':
        bins = np.arange(min_val, max_val, 0.05)

    hist_v, bins = np.histogram(characteristics_mat[~np.isnan(characteristics_mat[:, i]), i], bins=bins)
    hist, bins = np.histogram(characteristics_mat[~np.isnan(characteristics_mat[:, i]), i], bins=bins)

    ylim = 200
    dylim = 5
    if return_characteristics[i] == 'AP_width':
        ylim = int(np.ceil(np.max(hist_v))) + 5
        dylim = 10
    character_name_dict = {'AP_amp': 'AP amplitude (mV)', 'AP_width': 'AP width (ms)',
                           'fAHP_amp': 'fAHP amplitude (mV)',
                           'DAP_amp': 'DAP amplitude (mV)', 'DAP_deflection': 'DAP deflection (mV)',
                           'DAP_width': 'DAP width (ms)', 'DAP_time': 'DAP time (ms)'}

    fig, ax1 = pl.subplots()
    ax1.bar(bins[:-1], hist_v, width=bins[1] - bins[0], color='0.5')#, alpha=0.5)
    #ax1.set_ylim(0, ylim)
    #ax1.set_yticks(range(0, ylim, dylim))
    #ax2 = ax1.twinx()
    #ax2.bar(bins[:-1], hist, width=bins[1] - bins[0], color='r', alpha=0.5)
    #ax2.set_ylim(0, 4)
    #ax2.set_yticks(range(0, 4))
    ax1.set_xlabel(character_name_dict.get(return_characteristics[i], return_characteristics[i]))
    ax1.set_ylabel('Count')
    #ax2.set_ylabel('Count in vivo', fontsize=16)
    h1, l1 = ax1.get_legend_handles_labels()
    #h2, l2 = ax2.get_legend_handles_labels()
    #ax1.legend(h1 + h2, l1 + l2, fontsize=16)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_plots, 'hist_' + return_characteristics[i] + '.png'))
    pl.show()