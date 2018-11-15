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

        '$g_{Leak}$': ['soma', '0.5', 'pas', 'g'],
        'Nap $g_{max}$': ['soma', '0.5', 'nap', 'gbar'],
        'Nat $g_{max}$': ['soma', '0.5', 'nat', 'gbar'],
        'Kdr $g_{max}$': ['soma', '0.5', 'kdr', 'gbar'],
        'HCN $g_{max}$': ['soma', '0.5', 'hcn_slow', 'gbar'],

        'Nap $V_{h, m}$': ['soma', '0.5', 'nap', 'm_vh'],
        'Nap $V_{h, h}$': ['soma', '0.5', 'nap', 'h_vh'],
        'Nat $V_{h, m}$': ['soma', '0.5', 'nat', 'm_vh'],
        'Nat $V_{h, h}$': ['soma', '0.5', 'nat', 'h_vh'],
        'Kdr $V_{h, m}$': ['soma', '0.5', 'kdr', 'n_vh'],
        'HCN $V_{h, h}$': ['soma', '0.5', 'hcn_slow', 'n_vh'],

        'Nap $V_{s, m}$': ['soma', '0.5', 'nap', 'm_vs'],
        'Nap $V_{s, h}$': ['soma', '0.5', 'nap', 'h_vs'],
        'Nat $V_{s, m}$': ['soma', '0.5', 'nat', 'm_vs'],
        'Nat $V_{s, h}$': ['soma', '0.5', 'nat', 'h_vs'],
        'Kdr $V_{s, m}$': ['soma', '0.5', 'kdr', 'n_vs'],
        'HCN $V_{s, h}$': ['soma', '0.5', 'hcn_slow', 'n_vs'],

        'Nap $\\tau_{min, m}$': ['soma', '0.5', 'nap', 'm_tau_min'],
        'Nap $\\tau_{min, h}$': ['soma', '0.5', 'nap', 'h_tau_min'],
        'Nat $\\tau_{min, m}$': ['soma', '0.5', 'nat', 'm_tau_min'],
        'Nat $\\tau_{min, h}$': ['soma', '0.5', 'nat', 'h_tau_min'],
        'Kdr $\\tau_{min, m}$': ['soma', '0.5', 'kdr', 'n_tau_min'],
        'HCN $\\tau_{min, h}$': ['soma', '0.5', 'hcn_slow', 'n_tau_min'],

        'Nap $\\tau_{max, m}$': ['soma', '0.5', 'nap', 'm_tau_max'],
        'Nap $\\tau_{max, h}$': ['soma', '0.5', 'nap', 'h_tau_max'],
        'Nat $\\tau_{max, m}$': ['soma', '0.5', 'nat', 'm_tau_max'],
        'Nat $\\tau_{max, h}$': ['soma', '0.5', 'nat', 'h_tau_max'],
        'Kdr $\\tau_{max, m}$': ['soma', '0.5', 'kdr', 'n_tau_max'],
        'HCN $\\tau_{max, h}$': ['soma', '0.5', 'hcn_slow', 'n_tau_max'],

        'Nap $\\tau_{delta, m}$': ['soma', '0.5', 'nap', 'm_tau_delta'],
        'Nap $\\tau_{delta, h}$': ['soma', '0.5', 'nap', 'h_tau_delta'],
        'Nat $\\tau_{delta, m}$': ['soma', '0.5', 'nat', 'm_tau_delta'],
        'Nat $\\tau_{delta, h}$': ['soma', '0.5', 'nat', 'h_tau_delta'],
        'Kdr $\\tau_{delta, m}$': ['soma', '0.5', 'kdr', 'n_tau_delta'],
        'HCN $\\tau_{delta, h}$': ['soma', '0.5', 'hcn_slow', 'n_tau_delta'],
    }
    channels = ['Nat', 'Nap', 'Kdr', 'HCN']
    gates = ['m', 'h']
    params = ['$g_{max}$', '$V_h$', '$V_s$', '$\\tau_{min}$', '$\\tau_{max}$', '$\\tau_{delta}$']
    param_format = {
        '$c_m$': '%.2f',
        '$length$': '%.2f',
        '$diameter$': '%.2f',
        '$E_{Leak}$': '%.2f',
        '$E_{HCN}$': '%.2f',
        '$E_{Na}$': '%.2f',
        '$E_{K}$': '%.2f',
        '$g_{Leak}$': '%.5f',
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
        '$c_m$': '$\mu F/cm^2$',
        '$length$': '$\mu m$',
        '$diameter$': '$\mu m$',
        '$g_{Leak}$': '$S/cm^2$',
        '$E_{Leak}$': '$mV$',
        '$E_{HCN}$': '$mV$',
        '$E_{Na}$': '$mV$',
        '$E_{K}$': '$mV$'
    }

    param_val_dict = {k: cell.get_attr(v) for k, v in param_key_dict.iteritems()}

    table = []
    index = [p + ' $($' + param_unit[p] + '$)$' for p in params]
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
                elif param == '$V_h$' or param == '$V_s$' or param == '$\\tau_{min}$' or param == '$\\tau_{max}$' or param == '$\\tau_{delta}$':
                    param_ = param.replace('$', '').replace('{', '').replace('}', '')
                    p1, p2 = param_.split('_')
                    column[row] = param_format[param] % param_val_dict[channel + ' ' + '$'+p1+'_{'+p2+', '+gate+'}$']
                else:
                    column[row] = param_format[param] % param_val_dict[channel + ' ' + gate + ' ' + param]
            table.append(column)
    table = np.array(table).T

    header1 = r' & \multicolumn{2}{c}{$Na_T$} & \multicolumn{2}{c}{$Na_P$} & \multicolumn{1}{c}{$K_{DR}$} &  \multicolumn{1}{c}{$HCN$} \\'
    header2 = r' & \multicolumn{1}{c}{m} & \multicolumn{1}{c}{h} & \multicolumn{1}{c}{m} & \multicolumn{1}{c}{h} & \multicolumn{1}{c}{m} & \multicolumn{1}{c}{h} \\'
    line_break = r'\hline'
    powers = r' $p/q\ (1)$ & 3 & 1 & 3 & 1 & 4 & 1 \\'
    table = tabulate(table, tablefmt='latex_raw')

    table_lines = table.split('\n')
    table_lines.insert(1, line_break)
    table_lines.insert(2, header1)
    table_lines.insert(3, header2)
    table_lines.insert(5, powers)
    table = reduce(lambda a, b: a + '\n' + b, table_lines)
    print table

    other_params = ['$c_m$', '$length$', '$diameter$', '$g_{Leak}$', '$E_{Leak}$', '$E_{HCN}$', '$E_{Na}$', '$E_{K}$']
    index = [p + ' $($' + param_unit[p] + '$)$' for p in other_params]

    table2 = tabulate(np.array([index,
                                [param_val_dict[p] for p in other_params]]).T,
                      headers=['General Parameter'], tablefmt='latex_raw', floatfmt='.2f')
    print table2
    print 'g Leak: ', cell.soma(.5).pas.g