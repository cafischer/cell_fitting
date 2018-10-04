import os
import numpy as np
from tabulate import tabulate
from nrn_wrapper import Cell


if __name__ == '__main__':
    save_dir_table = '/home/cf/Phd/DAP-Project/thesis/tables'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    model = '2'

    # create model cell
    cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'), mechanism_dir)

    # save model parameters
    param_key_dict = {
        '$c_m$': ['soma', 'cm'],
        '$length$': ['soma', 'L'],
        '$diameter$': ['soma', 'diam'],
        '$E_{Leak}$': ['soma', '0.5', 'pas', 'e'],
        '$E_{HCN}$': ['soma', '0.5', 'hcn_slow', 'ehcn'],
        '$E_{Na}$': ['soma', '0.5', 'ena'],
        '$E_{K}$': ['soma', '0.5', 'ek'],

        'Leak $g_{max}$': ['soma', '0.5', 'pas', 'g'],
        'Nap $g_{max}$': ['soma', '0.5', 'nat', 'gbar'],
        'Nat $g_{max}$': ['soma', '0.5', 'nap', 'gbar'],
        'Kdr $g_{max}$': ['soma', '0.5', 'kdr', 'gbar'],
        'HCN $g_{max}$': ['soma', '0.5', 'hcn_slow', 'gbar'],

        'Nap m $V_h$': ['soma', '0.5', 'nat', 'm_vh'],
        'Nap h $V_h$': ['soma', '0.5', 'nat', 'h_vh'],
        'Nat m $V_h$': ['soma', '0.5', 'nap', 'm_vh'],
        'Nat h $V_h$': ['soma', '0.5', 'nap', 'h_vh'],
        'Kdr m $V_h$': ['soma', '0.5', 'kdr', 'n_vh'],
        'HCN h $V_h$': ['soma', '0.5', 'hcn_slow', 'n_vh'],

        'Nap m $V_s$': ['soma', '0.5', 'nat', 'm_vs'],
        'Nap h $V_s$': ['soma', '0.5', 'nat', 'h_vs'],
        'Nat m $V_s$': ['soma', '0.5', 'nap', 'm_vs'],
        'Nat h $V_s$': ['soma', '0.5', 'nap', 'h_vs'],
        'Kdr m $V_s$': ['soma', '0.5', 'kdr', 'n_vs'],
        'HCN h $V_s$': ['soma', '0.5', 'hcn_slow', 'n_vs'],

        'Nap m $\\tau_{min}$': ['soma', '0.5', 'nat', 'm_tau_min'],
        'Nap h $\\tau_{min}$': ['soma', '0.5', 'nat', 'h_tau_min'],
        'Nat m $\\tau_{min}$': ['soma', '0.5', 'nap', 'm_tau_min'],
        'Nat h $\\tau_{min}$': ['soma', '0.5', 'nap', 'h_tau_min'],
        'Kdr m $\\tau_{min}$': ['soma', '0.5', 'kdr', 'n_tau_min'],
        'HCN h $\\tau_{min}$': ['soma', '0.5', 'hcn_slow', 'n_tau_min'],

        'Nap m $\\tau_{max}$': ['soma', '0.5', 'nat', 'm_tau_max'],
        'Nap h $\\tau_{max}$': ['soma', '0.5', 'nat', 'h_tau_max'],
        'Nat m $\\tau_{max}$': ['soma', '0.5', 'nap', 'm_tau_max'],
        'Nat h $\\tau_{max}$': ['soma', '0.5', 'nap', 'h_tau_max'],
        'Kdr m $\\tau_{max}$': ['soma', '0.5', 'kdr', 'n_tau_max'],
        'HCN h $\\tau_{max}$': ['soma', '0.5', 'hcn_slow', 'n_tau_max'],

        'Nap m $\\tau_{delta}$': ['soma', '0.5', 'nat', 'm_tau_delta'],
        'Nap h $\\tau_{delta}$': ['soma', '0.5', 'nat', 'h_tau_delta'],
        'Nat m $\\tau_{delta}$': ['soma', '0.5', 'nap', 'm_tau_delta'],
        'Nat h $\\tau_{delta}$': ['soma', '0.5', 'nap', 'h_tau_delta'],
        'Kdr m $\\tau_{delta}$': ['soma', '0.5', 'kdr', 'n_tau_delta'],
        'HCN h $\\tau_{delta}$': ['soma', '0.5', 'hcn_slow', 'n_tau_delta'],
    }
    channels = ['Nat', 'Nap', 'Kdr', 'HCN']
    gates = ['m', 'h']
    params = ['$g_{max}$', '$V_h$', '$V_s$', '$\\tau_{min}$', '$\\tau_{max}$', '$\\tau_{delta}$']
    param_format = {
        '$g_{max}$': '%.5f',
        '$V_h$': '%.2f',
        '$V_s$': '%.2f',
        '$\\tau_{min}$': '%.3f',
        '$\\tau_{max}$': '%.3f',
        '$\\tau_{delta}$': '%.3f',
    }
    param_unit = {
        '$g_{max}$': '$S/cm^2$',
        '$V_h$': '$mV$',
        '$V_s$': '$mV$',
        '$\\tau_{min}$': '$ms$',
        '$\\tau_{max}$': '$ms$',
        '$\\tau_{delta}$': '$1$',
    }

    param_val_dict = {k: cell.get_attr(v) for k, v in param_key_dict.iteritems()}

    table = []
    index = [p + ' (' + param_unit[p] + ')' for p in params]
    table.append(index)
    for channel in channels:
        keys_channel = filter(lambda x: x.split(' ')[0] == channel, param_val_dict.keys())

        for gate in gates:
            if channel == 'Kdr' and gate == 'h' \
                    or channel == 'HCN' and gate == 'm':
                continue
            keys_gate = filter(lambda x: x.split(' ')[1] == gate, keys_channel)

            column = np.zeros(len(params))
            for row, param in enumerate(params):
                if param == '$g_{max}$':
                    column[row] = param_format[param] % param_val_dict[channel + ' ' + param]
                else:
                    column[row] = param_format[param] % param_val_dict[channel + ' ' + gate + ' ' + param]
            table.append(column)
    table = np.array(table).T

    header1 = ' & \multicolumn{2}{c}{Nat} & \multicolumn{2}{c}{Nap} & \multicolumn{1}{c}{Kdr} &  \multicolumn{1}{c}{HCN} \\'
    header2 = ' & \multicolumn{1}{c}{m} & \multicolumn{1}{c}{h} & \multicolumn{1}{c}{m} & \multicolumn{1}{c}{h} & \multicolumn{1}{c}{m} & \multicolumn{1}{c}{h} \\'
    line_break = r'\hline'
    table = tabulate(table, tablefmt='latex_raw')

    table_lines = table.split('\n')
    table_lines.insert(1, line_break)
    table_lines.insert(2, header1)
    table_lines.insert(3, header2)
    table = reduce(lambda a, b: a + '\n' + b, table_lines)
    print table

    other_params = ['$c_m$', '$length$', '$diameter$', '$E_{Leak}$', '$E_{HCN}$', '$E_{Na}$', '$E_{K}$']
    other_param_unit = {
        '$c_m$': '$\mu F/cm^2$',
        '$length$': '$\mu m$',
        '$diameter$': '$\mu m$',
        '$E_{Leak}$': '$mV$',
        '$E_{HCN}$': '$mV$',
        '$E_{Na}$': '$mV$',
        '$E_{K}$': '$mV$'
    }
    index = [p + ' (' + other_param_unit[p] + ')' for p in other_params]

    table2 = tabulate(np.array([index,
                                [param_val_dict[p] for p in other_params]]).T,
                      headers=['General Parameter'], tablefmt='latex_raw', floatfmt='.2f')
    print table2