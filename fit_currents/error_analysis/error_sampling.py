import pandas as pd
import numpy as np
from model_generator import *
from model.cell_builder import *
from optimization.fitfuns import run_simulation
from optimization.errfuns import rms
from optimization.optimizer import extract_simulation_params
from error_noiseV import *
from fit_currents.vclamp import *
from utilities import merge_dicts

__author__ = 'caro'


# parameter
n_models = 20
data_dir = '../../data/cell_2013_12_13f/dap/dap1.csv'
protocol = 'stepramp'
h.nrn_load_dll(complete_mechanismdir("../../model/channels_currentfitting"))
h.nrn_load_dll(complete_mechanismdir("../../model/channels_vclamp"))
all_channels = ['narsg', 'naf', 'na8st', 'nap', 'ka', 'kdr', 'ih_slow', 'ih_fast', 'passive']
E_ion = {'ena': 60, 'ek': -80, 'eca': 80, 'ehcn': -20, 'epas': -70}
all_channel_boundaries = [["g_narsg", 0, 0.1, [["soma", "mechanisms", "narsg", "gbar"]]],
                            ["g_naf", 0, 0.1, [["soma", "mechanisms", "naf", "gbar"]]],
                            ["g_na8st", 0, 0.1, [["soma", "mechanisms", "na8st", "gbar"]]],
                            ["g_nap", 0, 0.1, [["soma", "mechanisms", "nap", "gbar"]]],
                            ["g_ka", 0, 0.1, [["soma", "mechanisms", "ka", "gbar"]]],
                            ["g_kdr", 0, 0.1, [["soma", "mechanisms", "kdr", "gbar"]]],
                            ["g_ih_slow", 0, 0.1, [["soma", "mechanisms", "ih_slow", "gbar"]]],
                            ["g_ih_fast", 0, 0.1, [["soma", "mechanisms", "ih_fast", "gbar"]]],
                            ["g_passive", 0, 0.1, [["soma", "mechanisms", "passive", "gbar"]]]
                            ]

model_params = {
    "soma": {
        "geom": {
            "diam": 8.0,
            "L": 16.0
        },
        "Ra": 100,
        "cm": 1.0
    },
    "celsius": 35
}


# load experimental data
data = pd.read_csv(data_dir)
dt = data.t[1]-data.t[0]
dt_sim = 0.0025  # small as possible to reduce integration error
data = change_dt(dt_sim, data, protocol)
t = np.array(data.t)
i_inj = np.array(data.i)

# initialize errors
error_current = np.zeros([n_models], dtype=list)
error_baseline = np.zeros([n_models])
error_samplecurrent = np.zeros([n_models])
error_sampledvdt = np.zeros([n_models])
error_sampling = np.zeros([n_models])

# generate random models
for i in range(n_models):
    # randomly choose a set of channels
    mask = np.random.rand(len(all_channels)) > 0.5
    channel_list = np.array(all_channels)[mask]
    ion_list = get_ionlist(channel_list)
    channel_boundaries = np.array(all_channel_boundaries, dtype=object)[mask]

    model, vals = generate_model(channel_boundaries)
    model = merge_dicts(model, model_params)
    weights_model = {k: vals[j] for j, k in enumerate(channel_list)}

    # compute model response
    cell = Cell(model)
    cell_area = cell.soma(.5).area() * 1e-8
    cell = set_Epotential(cell, E_ion)
    simulation_params = extract_simulation_params(data)
    v_model, _ = run_simulation(cell, **simulation_params)
    v_model = np.array(v_model)
    dvdt_model = np.concatenate((np.array([(v_model[1]-v_model[0])/dt]), np.diff(v_model) / dt))
    #downsample = int(dt/dt_sim)
    downsample = 2

    # fit without downsampling
    currents_baseline = vclamp_withcurrents(v_model, t, Cell(model_params), channel_list, ion_list, E_ion)
    fit_baseline, _ = current_fitting(dvdt_model, t, i_inj, currents_baseline, cell_area, channel_list, cm=cell.soma.cm)

    error_tmp = [rms(weights_model[k], fit_baseline[k]) for k in weights_model.keys()]
    error_baseline[i] = np.mean(error_tmp)

    # error in ionic currents from sampling
    currents_sampled = vclamp_withcurrents(v_model[::downsample], t[::downsample], Cell(model_params),
                                          channel_list, ion_list, E_ion)
    #pl.figure()
    #pl.plot(t[::downsample], currents_baseline[0][::downsample], 'k', label='sampling rate: '+str(dt_sim))
    #pl.plot(t[::downsample], currents_sampled[0], 'b', label='sampling rate: '+str(dt))
    #pl.xlabel('Time $(ms)$')
    #pl.ylabel('Current $(mA/cm^2)$')
    #pl.legend()
    #pl.show()
    error_samplecurrent[i] = np.sum([rms(currents_baseline[j, ::downsample], currents_sampled[j])
                                     for j in range(len(currents_baseline))])

    #v_inter = np.interp(t, t[::downsample], v_model[::downsample])
    #currents_inter = vclamp_withcurrents(v_inter, t, Cell(model_params), channel_list, ion_list, E_ion)
    #pl.figure()
    #pl.plot(t, currents[0])
    #pl.plot(t, currents_vclamp[0])
    #pl.title('interpolate V')
    #pl.show()
    #print 'Error currents interpolating V: '+str(rms(currents_baseline[0], currents_inter[0]))

    # error in dV/dt from sampling
    #pl.figure()
    #pl.plot(t, v_model, 'k')
    #pl.plot(t[::downsample], v_model[::downsample], 'b')
    #pl.show()
    dvdt_sampled = np.concatenate((np.array([(v_model[1]-v_model[0])/dt]), np.diff(v_model[::downsample]) / dt))
    #pl.figure()
    #pl.plot(t, dvdt_model, 'k')
    #pl.plot(t[::downsample], dvdt_sampled, 'b')
    #pl.title('compare dV/dt')
    #pl.show()
    error_sampledvdt[i] = rms(dvdt_model[::downsample], dvdt_sampled)

    # error in fit from sampling
    fit_sampled, _ = current_fitting(dvdt_sampled, t[::downsample], i_inj[::downsample], currents_sampled, cell_area,
                                     channel_list, cm=cell.soma.cm)

    error_tmp = [rms(weights_model[k], fit_sampled[k]) for k in fit_sampled.keys()]
    error_sampling[i] = np.mean(error_tmp)



# compute mean error of all models
print 'Mean error from sampling V in current traces: ' + str(np.mean(error_samplecurrent))
print 'Mean error from sampling V in dV/dt: ' + str(np.mean(error_sampledvdt))

print 'Mean error in the weights with dV/dt estimated from V with a sampling rate of '+str(dt_sim)+': ' \
      + str(np.mean(error_baseline))
print 'Mean error in the weights with dV/dt estimated from V downsampled to a sampling rate of '+str(dt)+': ' \
      + str(np.mean(error_sampling))