import os
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
from neuron import h
from cell_fitting.optimization.simulate import iclamp_handling_onset, get_standard_simulation_params
from cell_fitting.read_heka import get_i_inj_from_function, get_sweep_index_for_amp
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from analyze_in_vivo.load.load_domnisoru import load_data
pl.style.use('paper_subplots')

if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # simulate with different g_Leak values
    #
    # g_pas_values = np.linspace(0.0004, 0.0005, 5)  # [0.00043, 0.00045, 0.0005, 0.00055, 0.00055]
    # colors = ['y', 'xkcd:orange', 'xkcd:red', 'm', 'b']
    #
    # # load model
    # cell = Cell.from_modeldir(model_dir, mechanism_dir)
    #
    # # get simulation_params
    # i_exp = get_i_inj_from_function('rampIV', [get_sweep_index_for_amp(3.1, 'rampIV')], 150, 0.01)[0]
    # simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -75, 'tstop': 150,
    #                      'dt': 0.01, 'celsius': 35, 'onset': 200}
    #
    # v_traces = []
    # DAP_deflections = np.zeros(len(g_pas_values))
    # for i, p_pas_value in enumerate(g_pas_values):
    #     # simulate
    #     cell.soma(.5).g_pas = p_pas_value
    #     v, t, _ = iclamp_handling_onset(cell, **simulation_params)
    #     v_traces.append(v)
    #
    #     # compute DAP deflection
    #     DAP_deflections[i] = get_spike_characteristics(v, t, ['DAP_deflection'], -75, **get_spike_characteristics_dict())[0]
    #
    # print 'DAP_deflections: ', DAP_deflections
    #
    # # plot
    # fig, ax = pl.subplots()
    # for i, p_pas_value in enumerate(g_pas_values):
    #     ax.plot(t, v_traces[i], label='$g_{Leak}$=%.6f' % p_pas_value)
    # ax.set_ylabel('Mem. pot. (mV)')
    # ax.set_xlabel('Time (ms)')
    # pl.legend()
    # pl.tight_layout()
    # #pl.show()

    # insert subthreshold mem. pot. from in vivo as g_Leak

    # parameters
    param_list = ['Vm_ljpc', 'Vm_wo_spikes_ljpc']
    resistance = 11.  # s73_0004: 10.  s104_0007: 12.
    add_factor = 0.75  # s73_0004: 0.78  s104_0007: 0.74

    # load data
    save_dir_data = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    data = load_data('s104_0007', param_list, save_dir_data)
    v = data['Vm_ljpc'][:1000000]
    v_sub = data['Vm_wo_spikes_ljpc'][:1000000]
    t = np.arange(0, len(v_sub)) * data['dt']

    # create cell model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    simulation_params = get_standard_simulation_params()
    simulation_params['tstop'] = t[-1]
    simulation_params['i_inj'] = np.zeros(len(t))

    # simulation
    v_sub = v_sub + np.min(v_sub)  # make positive
    g_Leak_trace = v_sub / np.max(v_sub) * cell.soma(.5).g_pas / 1.3
    g_Leak_vec = h.Vector()
    g_Leak_vec.from_python(g_Leak_trace)
    t_vec = h.Vector()
    t_vec.from_python(t)
    g_Leak_vec.play(cell.soma(.5)._ref_g_pas, t_vec, False)
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    # plot
    fig, ax = pl.subplots()
    ax.plot(t, v, 'k')
    ax.set_ylabel('Mem. pot. (mV)')
    ax.set_xlabel('Time (ms)')
    pl.tight_layout()
    pl.show()