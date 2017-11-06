import matplotlib.pyplot as pl
from scipy.stats import linregress, ttest_1samp
import numpy as np
import os
pl.style.use('paper')


# parameters
save_dir = './plots/hyperdap'
save_dir_summary = os.path.join(save_dir, 'summary_plots')

DAP_amps = np.load(os.path.join(save_dir_summary, 'DAP_amps.npy'))
DAP_deflections = np.load(os.path.join(save_dir_summary, 'DAP_deflections.npy'))
amps = np.load(os.path.join(save_dir_summary, 'amps.npy'))
cells = np.load(os.path.join(save_dir_summary, 'cells.npy'))

# # compute averages
# DAP_amp_mean = np.nanmean(DAP_amps, 0)
# DAP_amp_std = np.nanstd(DAP_amps, 0)
# DAP_deflection_mean = np.nanmean(DAP_deflections, 0)
# DAP_deflection_std = np.nanstd(DAP_deflections, 0)
#
# pl.figure()
# pl.errorbar(amps, DAP_amp_mean, DAP_amp_std, fmt='o', color='k', capsize=3)
# pl.xlabel('Current Amplitude (nA)')
# pl.ylabel('DAP Amplitude (mV)')
# pl.xticks(amps)
# pl.tight_layout()
# pl.savefig(os.path.join(save_dir_summary, 'DAP_amp.png'))
# # pl.show()
#
# pl.figure()
# pl.errorbar(amps, DAP_deflection_mean, DAP_deflection_std, fmt='o', color='k', capsize=3)
# pl.xlabel('Current Amplitude (nA)')
# pl.ylabel('DAP Deflection (mV)')
# pl.xticks(amps)
# pl.tight_layout()
# pl.savefig(os.path.join(save_dir_summary, 'DAP_deflection.png'))
# pl.show()

pl.figure()
cmap = pl.cm.get_cmap('jet')
colors = [cmap(x) for x in np.linspace(0, 1, len(DAP_amps))]
markers = [(2+i/2, 1+i%2, 0) for i in range(len(DAP_amps))]
for i, DAP_amps_cell in enumerate(DAP_amps):
    pl.plot(amps, DAP_amps_cell, color=colors[i], marker='o', markersize=10, label=cells[i])
#pl.legend()
pl.ylabel('DAP Amplitude (mV)')
pl.xlabel('Step Current Amplitude (nA)')
pl.tight_layout()
pl.savefig(os.path.join(save_dir_summary, 'DAP_amps_cells.png'))
pl.show()

slopes = np.zeros(len(DAP_amps))
intercepts = np.zeros(len(DAP_amps))
mse = np.zeros(len(DAP_amps))
for i, DAP_amps_cell in enumerate(DAP_amps):
    not_nan = ~np.isnan(DAP_amps_cell)
    slopes[i], intercepts[i], _, _, _ = linregress(amps[not_nan], DAP_amps_cell[not_nan])
    mse[i] = np.sqrt(np.mean(((amps[not_nan] * slopes[i] + intercepts[i]) - DAP_amps_cell[not_nan])**2))

# pl.figure()
# cmap = pl.cm.get_cmap('jet')
# colors = [cmap(x) for x in np.linspace(0, 1, len(DAP_amps))]
# markers = [(2 + i / 2, 1 + i % 2, 0) for i in range(len(DAP_amps))]
# for i, DAP_amps_cell in enumerate(DAP_amps):
#     pl.plot(amps, DAP_amps_cell, color=colors[i], marker='o', markersize=10, label=cells[i])
#     pl.plot(amps, amps * slopes[i] + intercepts[i], color=colors[i])
#pl.show()

print np.mean(slopes), np.std(slopes)
print np.mean(mse), np.std(mse)
t, p = ttest_1samp(slopes, 0)
print p, p < 0.01, p < 0.001, p < 0.0001

fig, ax = pl.subplots(1, 2)
ax[0].errorbar(0, np.mean(slopes), yerr=np.std(slopes), color='k', marker='o', capsize=3)
ax[0].errorbar(1, 0, yerr=0, color='k', marker='o')
ax[0].annotate('***', xy=(0.5, 0.75), arrowprops=dict(arrowstyle='-[, widthB=4.8, lengthB=0.4', lw=1.5), ha='center',
               fontsize=14)
ax[0].set_xticks([], [])
ax[0].set_ylabel('Slope (mV/nA)')
ax[0].set_ylim([-20, 1])
ax[1].errorbar(0, np.mean(mse), yerr=np.std(mse), color='k', marker='o', capsize=3)
ax[1].set_xticks([])
ax[1].set_ylabel('RMSE (mV)')
ax[1].set_ylim([0, 2])
pl.tight_layout()
pl.savefig(os.path.join(save_dir_summary, 'Slope_statistics.png'))
pl.show()