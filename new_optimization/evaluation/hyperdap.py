import pylab as pl
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm
from new_optimization.fitter.hodgkinhuxleyfitter import *
from new_optimization.evaluation.evaluate import *

__author__ = 'caro'


# parameters
save_dir = '../../results/new_optimization/2015_08_26b/22_01_17_readjust1/L-BFGS-B/'
hyperamps = np.arange(-0.9, 0.11, 0.2)
rampamp = 3.0
dt = 0.01

# load model
with open(save_dir + '/optimization_settings.json', 'r') as f:
    optimization_settings = json.load(f)
fitter = HodgkinHuxleyFitter(**optimization_settings['fitter'])
candidate = get_best_candidate(save_dir, n_best=1)
fitter.update_cell(candidate)

# construct current traces
hyp_st_ms = 4000.0*dt
hyp_end_ms = 16000.0*dt  # 12000
ramp_end_ms = 16400.0*dt  #12040.0*dt

t_exp = np.arange(0, hyp_st_ms+hyp_end_ms+ramp_end_ms, dt)

v = np.zeros([len(hyperamps), len(t_exp)])
for j, hyperamp in enumerate(hyperamps):
    i_exp = np.zeros(len(t_exp))
    i_exp[hyp_st_ms/dt:hyp_end_ms/dt] = hyperamp
    i_exp[hyp_end_ms/dt:hyp_end_ms/dt+(ramp_end_ms-hyp_end_ms)/dt/2] = np.linspace(hyperamp, rampamp,
                                                    len(i_exp[hyp_end_ms/dt:hyp_end_ms/dt+(ramp_end_ms-hyp_end_ms)/dt/2]))
    i_exp[hyp_end_ms/dt+(ramp_end_ms-hyp_end_ms)/dt/2:hyp_end_ms/dt] = np.linspace(rampamp, 0.0,
                                                    len(i_exp[hyp_end_ms/dt+(ramp_end_ms-hyp_end_ms)/dt/2:hyp_end_ms/dt]))

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -59, 'tstop': t_exp[-1],
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v[j], t, _ = iclamp_handling_onset(fitter.cell, **simulation_params)

# plot
pl.figure()
color = iter(cm.gist_rainbow(np.linspace(0, 1, len(hyperamps))))
for j, hyperamp in enumerate(hyperamps):
    pl.plot(t, v[j], c=next(color), label='amp: '+str(hyperamp))
pl.xlabel('Time $(ms)$', fontsize=16)
pl.ylabel('Membrane potential $(mV)$', fontsize=16)
pl.legend(loc='upper right', fontsize=16)
pl.show()