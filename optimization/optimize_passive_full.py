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

if __name__ == "__main__":

    # define objectives: zap and step current with different amplitudes
    objectives = ["zap"]
    i_steps = [-0.1, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.1]
    for i_step in i_steps:
        objectives.append('step_current_' + str(i_step))

    # specify the directory of the data for each objective
    data_dir = dict()
    data_dir[objectives[0]] = '../data/cell_2013_12_13f/zap_current/zap.csv'
    for objective in objectives[1:]:
        data_dir[objective] = '../data/cell_2013_12_13f/step_current/' + objective + '.csv'

    # variables that shall be optimized
    variables = [["cm", 0.7, 1.1, [["soma", "cm"], ["dendrites", "all", "cm"], ["axon_secs", "all", "cm"]]],
                ["Ra_soma", 10, 500,[["soma", "Ra"]]],
                ["Ra_dendrites", 10, 500, [["dendrites", "all", "Ra"]]],
                ["Ra_axon_secs", 10, 500, [["axon_secs", "all", "Ra"]]],
                ["g_pas", 1/1000, 1/100000, [["ion", "pas_ion", "g_pas"]]],
                ["e_pas", -65, -64, [["ion", "pas_ion", "e_pas"]]]]

    # create Optimizer
    optimizer = Optimizer(save_dir='./results_passive/StellateCell_full',
            data_dir=data_dir,
            model_dir='../model/cells/StellateCell_full.json', mechanism_dir=None,
            objectives=objectives,
            variables=variables,
            n_gen=1,
            emoo_params={'N': 1, 'C': 200, 'eta_m_0': 20, 'eta_c_0': 20, 'p_m': 0.5})

    complete_paths(variables)  # complete path specifications in variables

    # run and evaluate the evolution
    optimizer.run_emoo(optimizer.n_gen)
    optimizer.eval_emoo(10)


    # TODO: integrate complete paths
    # TODO: maybe set dendrite and axon params to a value before optimization because no experimental values to fit
    # TODO: maybe include h-current in model to better fit sag
    # TODO: Note: in Bahl only steady-state voltage is fit not whole current trace
