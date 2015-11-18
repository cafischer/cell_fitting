from __future__ import division
from optimizer import *
import copy

__author__ = 'caro'


def complete_paths(variables, cell):
    for variable in variables:
        paths = copy.deepcopy(variable[3])
        for path in paths:
            if "all" in path:
                variable[3].remove(path)
                for i in range(len(cell.params[path[0]])):
                    variable[3].append([path[0], str(i)] + path[2:])
    return variables

def impedance(v, i, dt, f_range):
    """
    Computes the impedance (impedance = fft(v) / fft(i)) for a given range of frequencies.

    :param v: Membrane potential (mV)
    :type v: array
    :param i: Current (nA)
    :type i: array
    :param dt: Time step.
    :type dt: float
    :param f_range: Boundaries of the frequency interval.
    :type f_range: list
    :return: Impedance (MOhm)
    :rtype: array
    """

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


def optimize_passive_full():

    # define objectives
    objectives = ['impedance', 'step_current_-0.1']

    # specify the directory of the data for each objective
    data_dir = dict()
    data_dir[objectives[0]] = '../data/cell_2013_12_13f/zap_current/impedance_ds.csv'
    data_dir[objectives[1]] =  '../data/cell_2013_12_13f/step_current/step_current_-0.1.csv'

    # variables that shall be optimized
    variables = [["cm", 0.6, 1.1, [["soma", "cm"], ["dendrites", "all", "cm"], ["axon_secs", "all", "cm"]]],
                ["Ra_soma", 10, 500,[["soma", "Ra"]]],
                ["g_pas", 1/100, 1/100000, [["ion", "pas", "g_pas"]]],  # g_pas: 1/1000, 1/100000
                ["e_pas", -80, -50, [["ion", "pas", "e_pas"]]],
                ["gfastbar", 0.0, 0.01, [["soma", "mechanisms", "ih", "gfastbar"],
                                         ["dendrites", "all", "mechanisms", "ih", "gfastbar"]]],  # insert h-current
                ["gslowbar", 0.0, 0.01, [["soma", "mechanisms", "ih", "gslowbar"],
                                         ["dendrites", "all", "mechanisms", "ih", "gslowbar"]]],
                ["g", 0.0, 0.01, [["soma", "mechanisms", "kleak", "g"],
                                         ["dendrites", "all", "mechanisms", "kleak", "g"]]]
                ]

    # create Optimizer
    optimizer = Optimizer(save_dir='./results_passive/StellateCell_full_min',
            data_dir=data_dir,
            model_dir='../model/cells/StellateCell_full_ih.json',
                          mechanism_dir='../model/channels/i686/.libs/libnrnmech.so',
            objectives=objectives,
            variables=variables,
            n_gen=25,
            emoo_params={'N': 100, 'C': 150, 'eta_m_0': 20, 'eta_c_0': 20, 'p_m': 0.5},
            get_var_to_fit=None,
            var_to_fit={'impedance': 'impedance', 'step_current_-0.1': 'v'})

    # functions to optimize
    def get_var1_to_fit(sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        f_range = [optimizer.data['impedance'].f_range[0], optimizer.data['impedance'].f_range[1]]
        v, t, i = optimizer.run_simulation(sec, i_amp, v_init, tstop, dt, pos_i, pos_v)
        imp, freqs = impedance(v, i, dt/1000, f_range)
        return imp, freqs

    def get_var2_to_fit(sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        v, t, i = optimizer.run_simulation(sec, i_amp, v_init, tstop, dt, pos_i, pos_v)
        return v, t

    get_var_to_fit = dict()
    get_var_to_fit['impedance'] = get_var1_to_fit
    get_var_to_fit['step_current_-0.1'] = get_var2_to_fit
    optimizer.get_var_to_fit = get_var_to_fit

    optimizer.variables = complete_paths(variables, optimizer.cell)  # complete path specifications in variables

    return optimizer


if __name__ == "__main__":
    # load skript
    optimizer = optimize_passive_full()

    # run and evaluate the evolution
    optimizer.run_emoo(optimizer.n_gen)
    optimizer.eval_emoo(10)

    # TODO: maybe divide between simulator (new class) and optimizer
    # TODO: integrate complete paths


    """
    # insert h-current in the model Cell
    h.nrn_load_dll('../model/channels/i686/.libs/libnrnmech.so')
    optimizer.cell.update_attr(['soma', 'mechanisms', 'ih'], {})
    for var in complete_paths([['ih', 0.0, 0.01, [['dendrites', 'all', 'mechanisms', 'ih']]]], optimizer.cell):
            for path in var[3]:
                optimizer.cell.update_attr(path, {})
    optimizer.cell.save_as_json('../model/cells/StellateCell_full_ih.json')
    """