from __future__ import division
from optimizer import *
from optimize_passive_full import impedance, complete_paths

__author__ = 'caro'


def optimize_passive_point():
    # define objectives
    objectives = ['impedance', 'step_current_-0.1']

    # specify the directory of the data for each objective
    data_dir = dict()
    data_dir[objectives[0]] = '../data/cell_2013_12_13f/zap_current/impedance_ds.csv'
    data_dir[objectives[1]] =  '../data/cell_2013_12_13f/step_current/step_current_-0.1.csv'

    # variables that shall be optimized
    variables = [["cm", 0.6, 1.1, [["soma", "cm"]]],
                ["L", 10, 100, [["soma", "geom", "L"]]],
                ["diam", 5, 100, [["soma", "geom", "diam"]]],
                ["Ra_soma", 10, 400,[["soma", "Ra"]]],
                ["g_pas", 1/10, 1/100000, [["ion", "pas", "g_pas"]]],  # g_pas: 1/1000, 1/100000
                ["e_pas", -120, -50, [["ion", "pas", "e_pas"]]],
                ["gfastbar", 0.0, 0.1, [["soma", "mechanisms", "ih", "gfastbar"]]],  # insert h-current
                ["gslowbar", 0.0, 0.1, [["soma", "mechanisms", "ih", "gslowbar"]]],
                ["g", 0.0, 0.01, [["soma", "mechanisms", "kleak", "g"]]],
                ["gnapbar", 0.0, 0.01, [["soma", "mechanisms", "nap", "gnapbar"]]]
                ]

    optimizer = Optimizer(save_dir='./results_passive/point_ihkleaknap3',
            data_dir=data_dir,
            model_dir='../model/cells/StellateCell_point.json',
            mechanism_dir='../model/channels/i686/.libs/libnrnmech.so',
            objectives=objectives,
            variables=variables,
            n_gen=50,
            emoo_params={'N': 100, 'C': 150, 'eta_m_0': 10, 'eta_c_0': 10, 'p_m': 0.75},
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
    optimizer = optimize_passive_point()

    optimizer.run_emoo(optimizer.n_gen)
    optimizer.eval_emoo(100)