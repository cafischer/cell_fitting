from __future__ import division
from optimizer import *

__author__ = 'caro'


def optimize_passive_point():
    # define objectives
    objectives = ['step_current_-0.1', 'impedance']

    # specify the directory of the data for each objective
    data_dir = dict()
    data_dir[objectives[0]] = '../data/cell_2013_12_13f/step_current/step_current_-0.1.csv'
    data_dir[objectives[1]] = '../data/cell_2013_12_13f/zap_current/impedance_ds.csv'

    # variables that shall be optimized
    variables = [["cm", 1.0, 2.0, [["soma", "cm"]]],
                 ["L", 80, 100, [["soma", "geom", "L"]]],
                 ["diam", 60, 100, [["soma", "geom", "diam"]]],
                 ["g_pas", 1/50000, 1/200000, [["ion", "pas", "g_pas"]]],  # g_pas: 1/1000, 1/100000
                 ["e_pas", -100, -60, [["ion", "pas", "e_pas"]]],
                 ["gfastbar", 0.0, 0.0001, [["soma", "mechanisms", "ih", "gfastbar"]]],
                 ["gslowbar", 0.0, 0.0001, [["soma", "mechanisms", "ih", "gslowbar"]]],
                 ["gkleakbar", 0.0, 0.0001, [["soma", "mechanisms", "kleak", "gkleakbar"]]],
                 ["ehcn", -30, -10, [["soma", "mechanisms", "ih", "ehcn"]]],
                 ["ekleak", -110, -80, [["soma", "mechanisms", "kleak", "ekleak"]]]
                 ]

    # functions to be optimized
    def fun_to_fit1(optimizer, sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        v, t, i = optimizer.run_simulation(sec, i_amp, v_init, tstop, dt, pos_i, pos_v)
        return v, t

    def fun_to_fit2(optimizer, sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        f_range = [optimizer.data['impedance'].f_range[0], optimizer.data['impedance'].f_range[1]]
        v, t, i = optimizer.run_simulation(sec, i_amp, v_init, tstop, dt, pos_i, pos_v)
        imp, freqs = impedance(v, i, dt/1000, f_range)
        return imp, freqs

    fun_to_fit = dict()
    fun_to_fit['step_current_-0.1'] = fun_to_fit1
    fun_to_fit['impedance'] = fun_to_fit2

    optimizer = Optimizer(
        save_dir='./results_passive/point/step_current_-0.1_impedance',
        data_dir=data_dir,
        model_dir='../model/cells/StellateCell_point.json',
        mechanism_dir='../model/channels/x86_64/.libs/libnrnmech.so',
        objectives=objectives,
        variables=variables,
        n_gen=200,
        emoo_params={'N': 1000, 'C': 1500, 'eta_m_0': 7, 'eta_c_0': 7, 'p_m': 0.25, 'd_eta_m': 0.5, 'd_eta_c': 0.5},
        fun_to_fit=fun_to_fit,
        var_to_fit={'impedance': 'impedance', 'step_current_-0.1': 'v'})

    return optimizer


def optimize_passive_point2():
    # define objectives
    objectives = ['step_current1_-0.1', 'step_current2_-0.1', 'step_current3_-0.1']

    # specify the directory of the data for each objective
    data_dir = dict()
    data_dir[objectives[0]] = '../data/cell_2013_12_13f/step_current/split/step_current_-0.1.csv'
    data_dir[objectives[1]] = '../data/cell_2013_12_13f/step_current/split/step_current_-0.1.csv'
    data_dir[objectives[2]] = '../data/cell_2013_12_13f/step_current/split/step_current_-0.1.csv'

    # variables that shall be optimized
    variables = [["cm", 1.0, 2.0, [["soma", "cm"]]],
                 ["L", 80, 100, [["soma", "geom", "L"]]],
                 ["diam", 80, 100, [["soma", "geom", "diam"]]],
                 ["g_pas", 1/50000, 1/200000, [["ion", "pas", "g_pas"]]],  # g_pas: 1/1000, 1/100000
                 ["e_pas", -100, -60, [["ion", "pas", "e_pas"]]],
                 ["gfastbar", 0.0, 0.0001, [["soma", "mechanisms", "ih", "gfastbar"]]],
                 ["gslowbar", 0.0, 0.0001, [["soma", "mechanisms", "ih", "gslowbar"]]],
                 ["gkleakbar", 0.0, 0.0001, [["soma", "mechanisms", "kleak", "gkleakbar"]]],
                 ["gnapbar", 0.0, 0.0001, [["soma", "mechanisms", "nap", "gnapbar"]]],
                 ["ehcn", -30, -10, [["soma", "mechanisms", "ih", "ehcn"]]],
                 ["ekleak", -110, -80, [["soma", "mechanisms", "kleak", "ekleak"]]],
                 ["ena", 80, 100, [["ion", "nap", "ena"]]]
                 ]

    # var_to_fit
    var_to_fit = {'step_current1_-0.1': 'v1', 'step_current2_-0.1': 'v2', 'step_current3_-0.1': 'v3'}

    # functions to fit
    def fun_to_fit1(sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        i_amp = list(i_amp)
        i_start = i_amp.index(filter(lambda x: x!=0, i_amp)[0])
        v, t, i = optimizer.run_simulation(sec, i_amp, v_init, tstop, dt, pos_i, pos_v)
        return v[:i_start], t[:i_start]

    def fun_to_fit2(sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        i_amp = list(i_amp)
        i_start = i_amp.index(filter(lambda x: x!=0, i_amp)[0])
        i_end = i_start + i_amp[i_start:].index(filter(lambda x: x==0, i_amp[i_start:])[0])
        v, t, i = optimizer.run_simulation(sec, i_amp, v_init, tstop, dt, pos_i, pos_v)
        return v[i_start:i_end], t[i_start:i_end]

    def fun_to_fit3(sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        i_amp = list(i_amp)
        i_start = i_amp.index(filter(lambda x: x!=0, i_amp)[0])
        i_end = i_start + i_amp[i_start:].index(filter(lambda x: x==0, i_amp[i_start:])[0])
        v, t, i = optimizer.run_simulation(sec, i_amp, v_init, tstop, dt, pos_i, pos_v)
        return v[i_end:], t[i_end:]

    fun_to_fit = dict()
    fun_to_fit['step_current1_-0.1'] = fun_to_fit1
    fun_to_fit['step_current2_-0.1'] = fun_to_fit2
    fun_to_fit['step_current3_-0.1'] = fun_to_fit3

    optimizer = Optimizer(save_dir='./results_passive/point_ihkleaknap_step3',
            data_dir=data_dir,
            model_dir='../model/cells/StellateCell_point.json',
            mechanism_dir='../model/channels/i686/.libs/libnrnmech.so',
            objectives=objectives,
            variables=variables,
            n_gen=5,
            emoo_params={'N': 2, 'C': 700, 'eta_m_0': 10, 'eta_c_0': 5, 'p_m': 0.5, 'd_eta_m': 1, 'd_eta_c': 1},
            get_var_to_fit=None,
            var_to_fit=var_to_fit)

    return optimizer

if __name__ == "__main__":
    optimizer = optimize_passive_point()
    optimizer.run_emoo()
    optimizer.eval_emoo(100)