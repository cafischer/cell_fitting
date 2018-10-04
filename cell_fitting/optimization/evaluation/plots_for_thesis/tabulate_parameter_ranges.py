import os
import numpy as np
from tabulate import tabulate
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys


if __name__ == '__main__':
    save_dir_table = '/home/cf/Phd/DAP-Project/thesis/tables'

    # variables
    variables = [
        [0.3, 2, '$c_m$'],
        [-100, -75, '$E_{Leak}$'],
        [-30, -10, '$E_{HCN}$'],

        [0, 0.5, '$g_{max}$'],

        [-100, 0, '$m$ $V_h$'],
        [-100, 0, '$h$ $V_h$'],

        [1, 30, '$m$ $V_s$'],
        [-30, -1, '$h$ $V_s$'],

        [0, 50, '$m$ $\\tau_{min}$'],
        [0, 50, '$h$ $\\tau_{min}$'],

        [0, 100, '$m$ $\\tau_{max}$'],
        [0, 100, '$h$ $\\tau_{max}$'],
        [0, 500, '$HCN$ $h$ $\\tau_{max}$'],

        [0, 1, '$m$ $\\tau_{delta}$'],
        [0, 1, '$h$ $\\tau_{delta}$'],
    ]

    lower_bound, upper_bound, variable_names = get_lowerbound_upperbound_keys(variables)

    param_format = {
        '$c_m$': '%.2f',
        '$length$': '%.2f',
        '$diameter$': '%.2f',
        '$E_{Leak}$': '%.2f',
        '$E_{HCN}$': '%.2f',
        '$E_{Na}$': '%.2f',
        '$E_{K}$': '%.2f',
        '$g_{max}$': '%.5f',
        '$V_h$': '%.2f',
        '$V_s$': '%.2f',
        '$\\tau_{min}$': '%.3f',
        '$\\tau_{max}$': '%.3f',
        '$\\tau_{delta}$': '%.3f',
    }
    param_unit = {
        '$c_m$': '$\mu F/cm^2$',
        '$length$': '$\mu m$',
        '$diameter$': '$\mu m$',
        '$E_{Leak}$': '$mV$',
        '$E_{HCN}$': '$mV$',
        '$E_{Na}$': '$mV$',
        '$E_{K}$': '$mV$',
        '$g_{max}$': '$S/cm^2$',
        '$V_h$': '$mV$',
        '$V_s$': '$mV$',
        '$\\tau_{min}$': '$ms$',
        '$\\tau_{max}$': '$ms$',
        '$\\tau_{delta}$': '$1$',
    }

    table = []
    for lb, ub, param in zip(lower_bound, upper_bound, variable_names):
        split = param.split(' ')
        p = split[0] if len(split) == 1 else split[-1]
        table.append([param + ' (' + param_unit[p] + ')', lb, ub])

    header = '\multicolumn{1}{c}{Parameter} & \multicolumn{1}{c}{Lower bound} & \multicolumn{1}{c}{Upper bound} \\'
    line_break = r'\hline'
    table = tabulate(table, tablefmt='latex_raw')

    table_lines = table.split('\n')
    table_lines.insert(1, line_break)
    table_lines.insert(2, header)
    table = reduce(lambda a, b: a + '\n' + b, table_lines)
    print table