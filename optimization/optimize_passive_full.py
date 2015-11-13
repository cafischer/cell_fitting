from __future__ import division
from optimizer import *
import copy

__author__ = 'caro'


def complete_paths(variables):
    for variable in variables:
        paths = copy.deepcopy(variable[3])
        for path in paths:
            if "all" in path:
                variable[3].remove(path)
                for i in range(len(optimizer.cell.params[path[0]])):
                    variable[3].append([path[0], str(i)] + path[2:])


def impedance(v, i, dt, f_range):
    # FFT of the membrance potential and the input current
    fft_i = np.fft.fft(i)
    fft_v = np.fft.fft(v)
    freqs = np.fft.fftfreq(v.size, d=dt)

    # sort everything according to the frequencies
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fft_i = fft_i[idx]
    fft_v = fft_v[idx]

    # calculate the impedance
    imp = np.abs(fft_v/fft_i)

    # index with frequency range
    idx1 = np.argmin(np.abs(freqs-f_range[0]))
    idx2 = np.argmin(np.abs(freqs-f_range[1]))

    return imp[idx1:idx2], freqs[idx1:idx2]


if __name__ == "__main__":

    # define objectives
    objectives = ["impedance"]
    i_steps = [-0.1]
    for i_step in i_steps:
        objectives.append('step_current_' + str(i_step))

    # specify the directory of the data for each objective
    data_dir = dict()
    data_dir[objectives[0]] = '../data/cell_2013_12_13f/zap_current/impedance.csv'
    for objective in objectives[1:]:
        data_dir[objective] = '../data/cell_2013_12_13f/step_current/' + objective + '.csv'

    # variables that shall be optimized
    variables = [["cm", 0.7, 1.1, [["soma", "cm"], ["dendrites", "all", "cm"], ["axon_secs", "all", "cm"]]],
                ["Ra_soma", 10, 500,[["soma", "Ra"]]],
                ["g_pas", 1/100, 1/10000, [["ion", "pas", "g_pas"]]],
                ["e_pas", -65, -64, [["ion", "pas", "e_pas"]]]]  # g_pas: 1/1000, 1/100000

    #  ["Ra_dendrites", 10, 500, [["dendrites", "all", "Ra"]]],
    #  ["Ra_axon_secs", 10, 500, [["axon_secs", "all", "Ra"]]],


    # create Optimizer
    optimizer = Optimizer(save_dir='./results_passive/StellateCell_full',
            data_dir=data_dir,
            model_dir='../model/cells/StellateCell_full.json', mechanism_dir=None,
            objectives=objectives,
            variables=variables,
            n_gen=50,
            emoo_params={'N': 100, 'C': 150, 'eta_m_0': 20, 'eta_c_0': 20, 'p_m': 0.5},
            get_var_to_fit=None,
            var_to_fit={'impedance': 'impedance', 'step_current_-0.1': 'v'})


    # run simulation
    def get_var1_to_fit(sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        f_range = [optimizer.data['impedance'].f_range[0], optimizer.data['impedance'].f_range[1]]
        v, t, i = optimizer.run_simulation(sec, i_amp, v_init, tstop, dt, pos_i, pos_v)
        imp, _ = impedance(v, i, dt/1000, f_range)
        return imp

    def get_var2_to_fit(sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        v, t, i = optimizer.run_simulation(sec, i_amp, v_init, tstop, dt, pos_i, pos_v)
        return v

    get_var_to_fit = dict()
    get_var_to_fit['impedance'] = get_var1_to_fit
    get_var_to_fit['step_current_-0.1'] = get_var2_to_fit
    optimizer.get_var_to_fit = get_var_to_fit

    complete_paths(variables)  # complete path specifications in variables

    # run and evaluate the evolution
    optimizer.run_emoo(optimizer.n_gen)
    optimizer.eval_emoo(10)

    # TODO: maybe divide between simulator (new class) and optimizer

    # TODO: integrate complete paths

    # TODO: set Ra of dendrites and axon_secs beforehand
    # TODO: maybe just use one current step: in (Destexhe, 2001) they use -0.1 nA current pulse
