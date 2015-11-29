from __future__ import division
from optimizer import *

__author__ = 'caro'

def optimize_passive_full():

    # define objectives
    objectives = ['impedance', 'step_current_-0.1']

    # specify the directory of the data for each objective
    data_dir = dict()
    data_dir[objectives[0]] = '../data/cell_2013_12_13f/zap_current/impedance_ds.csv'
    data_dir[objectives[1]] =  '../data/cell_2013_12_13f/step_current/step_current_-0.1.csv'

    # variables to optimize
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

    # functions to optimize
    def fun_to_fit1(sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        v, t, i = optimizer.run_simulation(sec, i_amp, v_init, tstop, dt, pos_i, pos_v)
        return v, t

    def fun_to_fit2(sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        f_range = [optimizer.data['impedance'].f_range[0], optimizer.data['impedance'].f_range[1]]
        v, t, i = optimizer.run_simulation(sec, i_amp, v_init, tstop, dt, pos_i, pos_v)
        imp, freqs = impedance(v, i, dt/1000, f_range)
        return imp, freqs

    fun_to_fit = dict()
    fun_to_fit['step_current_-0.1'] = fun_to_fit1
    fun_to_fit['impedance'] = fun_to_fit2

    # create Optimizer
    optimizer = Optimizer(save_dir='./results_passive/StellateCell_full_ihkleaknap1',
            data_dir=data_dir,
            model_dir='../model/cells/StellateCell_full_ih.json',
            mechanism_dir='../model/channels/i686/.libs/libnrnmech.so',
            objectives=objectives,
            variables=variables,
            n_gen=25,
            emoo_params={'N': 100, 'C': 150, 'eta_m_0': 20, 'eta_c_0': 20, 'p_m': 0.5},
            fun_to_fit=fun_to_fit,
            var_to_fit={'impedance': 'impedance', 'step_current_-0.1': 'v'})

    return optimizer


if __name__ == "__main__":
    optimizer = optimize_passive_full()
    optimizer.run_emoo()
    optimizer.eval_emoo(10)