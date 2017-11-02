import matplotlib.pyplot as pl
import numpy as np
import os
pl.style.use('paper')


# parameters
save_dir = './plots/hyperdap'
save_dir_summary = os.path.join(save_dir, 'summary_plots')

DAP_amps = np.load(os.path.join(save_dir_summary, 'DAP_amps.npy'))
DAP_deflections = np.load(os.path.join(save_dir_summary, 'DAP_deflections.npy'))
amps = np.load(os.path.join(save_dir_summary, 'amps.npy'))

# compute averages
DAP_amp_mean = np.nanmean(DAP_amps, 0)
DAP_amp_std = np.nanstd(DAP_amps, 0)
DAP_deflection_mean = np.nanmean(DAP_deflections, 0)
DAP_deflection_std = np.nanstd(DAP_deflections, 0)

pl.figure()
pl.errorbar(amps, DAP_amp_mean, DAP_amp_std, fmt='o', color='k', capsize=3)
pl.xlabel('Current Amplitude (nA)')
pl.ylabel('DAP Amplitude (mV)')
pl.xticks(amps)
pl.tight_layout()
pl.savefig(os.path.join(save_dir_summary, 'DAP_amp.png'))
# pl.show()

pl.figure()
pl.errorbar(amps, DAP_deflection_mean, DAP_deflection_std, fmt='o', color='k', capsize=3)
pl.xlabel('Current Amplitude (nA)')
pl.ylabel('DAP Deflection (mV)')
pl.xticks(amps)
pl.tight_layout()
pl.savefig(os.path.join(save_dir_summary, 'DAP_deflection.png'))
pl.show()


# TODO: fit straight line and plot average over that