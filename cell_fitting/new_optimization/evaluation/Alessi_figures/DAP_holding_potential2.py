import numpy as np
import os
from cell_characteristics.analyze_APs import get_spike_characteristics
import matplotlib.pyplot as pl
pl.style.use('paper')


# parameter
save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
save_img = os.path.join(save_dir, 'img', 'DAP_at_different_holding_potentials')

# parameters
return_characteristics = ['DAP_amp', 'DAP_deflection']

# load membrane potentials
v_mat = np.load(os.path.join(save_img, 'v_mat.npy'))
t = np.load(os.path.join(save_img, 't.npy'))
holding_potentials = np.load(os.path.join(save_img, 'hold_potentials.npy'))

# get DAP characteristics
DAP_amps = np.zeros(len(holding_potentials))
DAP_deflections = np.zeros(len(holding_potentials))
for i, v in enumerate(v_mat):
    DAP_amps[i], DAP_deflections[i] = get_spike_characteristics(v, t, return_characteristics, holding_potentials[i], AP_interval=4,
                                                std_idx_times=(0, 50), k_splines=5, s_splines=0, order_fAHP_min=None,
                                                DAP_interval=40, order_DAP_max=None, min_dist_to_DAP_max=0, check=True)
    print DAP_amps[i], DAP_deflections[i]

# plot
save_img = os.path.join(save_dir, 'img', 'DAP_at_different_holding_potentials')
if not os.path.exists(save_img):
    os.makedirs(save_img)

pl.figure()
pl.plot(holding_potentials, DAP_deflections, 'o-r')
pl.xlabel('Holding Potential (mV)')
pl.ylabel('DAP Deflection (mV)')
pl.xticks(holding_potentials)
pl.tight_layout()
pl.savefig(os.path.join(save_img, 'DAP_deflection.png'))
pl.show()

pl.figure()
pl.plot(holding_potentials, DAP_amps, 'o-r')
pl.xlabel('Holding Potential (mV)')
pl.ylabel('DAP amplitude (mV)')
pl.xticks(holding_potentials)
pl.tight_layout()
pl.savefig(os.path.join(save_img, 'DAP_amplitude.png'))
pl.show()

pl.figure()
pl.plot(holding_potentials, DAP_amps+holding_potentials, 'o-r')
pl.xlabel('Holding Potential (mV)')
pl.ylabel('DAP absolute level at peak (mV)')
pl.xticks(holding_potentials)
pl.tight_layout()
pl.savefig(os.path.join(save_img, 'DAP_abs_level_peak.png'))
pl.show()



# TODO: other approximation of fAHPmin?? if just shoulder no deflection


