import os
import numpy as np
from tabulate import tabulate
from nrn_wrapper import Cell, load_mechanism_dir
import collections
from cell_fitting.sensitivity_analysis.create_variable_range import get_order_of_magnitude


if __name__ == '__main__':
    save_dir_table = '/home/cf/Phd/DAP-Project/thesis/tables'
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    models = ['2', '3', '4', '5', '6']
    load_mechanism_dir(mechanism_dir)

    # save model parameters
    param_key_tuples = [
        ('$c_m$', ['soma', 'cm']),
        ('$length$', ['soma', 'L']),
        ('$diameter$', ['soma', 'diam']),
        ('$E_{Leak}$', ['soma', '0.5', 'pas', 'e']),
        ('$E_{HCN}$', ['soma', '0.5', 'hcn_slow', 'ehcn']),
        ('$E_{Na}$', ['soma', '0.5', 'ena']),
        ('$E_{K}$', ['soma', '0.5', 'ek']),

        ('$g_{Leak}$', ['soma', '0.5', 'pas', 'g']),
        ('Nap $g_{max}$', ['soma', '0.5', 'nap', 'gbar']),
        ('Nat $g_{max}$', ['soma', '0.5', 'nat', 'gbar']),
        ('Kdr $g_{max}$', ['soma', '0.5', 'kdr', 'gbar']),
        ('HCN $g_{max}$', ['soma', '0.5', 'hcn_slow', 'gbar']),

        ('Nap $V_{h, m}$', ['soma', '0.5', 'nap', 'm_vh']),
        ('Nap $V_{h, h}$', ['soma', '0.5', 'nap', 'h_vh']),
        ('Nat $V_{h, m}$', ['soma', '0.5', 'nat', 'm_vh']),
        ('Nat $V_{h, h}$', ['soma', '0.5', 'nat', 'h_vh']),
        ('Kdr $V_{h, m}$', ['soma', '0.5', 'kdr', 'n_vh']),
        ('HCN $V_{h, h}$', ['soma', '0.5', 'hcn_slow', 'n_vh']),

        ('Nap $V_{s, m}$', ['soma', '0.5', 'nap', 'm_vs']),
        ('Nap $V_{s, h}$', ['soma', '0.5', 'nap', 'h_vs']),
        ('Nat $V_{s, m}$', ['soma', '0.5', 'nat', 'm_vs']),
        ('Nat $V_{s, h}$', ['soma', '0.5', 'nat', 'h_vs']),
        ('Kdr $V_{s, m}$', ['soma', '0.5', 'kdr', 'n_vs']),
        ('HCN $V_{s, h}$', ['soma', '0.5', 'hcn_slow', 'n_vs']),

        ('Nap $\\tau_{min, m}$', ['soma', '0.5', 'nap', 'm_tau_min']),
        ('Nap $\\tau_{min, h}$', ['soma', '0.5', 'nap', 'h_tau_min']),
        ('Nat $\\tau_{min, m}$', ['soma', '0.5', 'nat', 'm_tau_min']),
        ('Nat $\\tau_{min, h}$', ['soma', '0.5', 'nat', 'h_tau_min']),
        ('Kdr $\\tau_{min, m}$', ['soma', '0.5', 'kdr', 'n_tau_min']),
        ('HCN $\\tau_{min, h}$', ['soma', '0.5', 'hcn_slow', 'n_tau_min']),

        ('Nap $\\tau_{max, m}$', ['soma', '0.5', 'nap', 'm_tau_max']),
        ('Nap $\\tau_{max, h}$', ['soma', '0.5', 'nap', 'h_tau_max']),
        ('Nat $\\tau_{max, m}$', ['soma', '0.5', 'nat', 'm_tau_max']),
        ('Nat $\\tau_{max, h}$', ['soma', '0.5', 'nat', 'h_tau_max']),
        ('Kdr $\\tau_{max, m}$', ['soma', '0.5', 'kdr', 'n_tau_max']),
        ('HCN $\\tau_{max, h}$', ['soma', '0.5', 'hcn_slow', 'n_tau_max']),

        ('Nap $\\tau_{delta, m}$', ['soma', '0.5', 'nap', 'm_tau_delta']),
        ('Nap $\\tau_{delta, h}$', ['soma', '0.5', 'nap', 'h_tau_delta']),
        ('Nat $\\tau_{delta, m}$', ['soma', '0.5', 'nat', 'm_tau_delta']),
        ('Nat $\\tau_{delta, h}$', ['soma', '0.5', 'nat', 'h_tau_delta']),
        ('Kdr $\\tau_{delta, m}$', ['soma', '0.5', 'kdr', 'n_tau_delta']),
        ('HCN $\\tau_{delta, h}$', ['soma', '0.5', 'hcn_slow', 'n_tau_delta']),
    ]
    param_key_dict = collections.OrderedDict(param_key_tuples)

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

    index = param_key_dict.keys()
    table_body = np.zeros((len(index), len(models)), dtype=object)
    for model_idx, model in enumerate(models):
        cell = Cell.from_modeldir(os.path.join(save_dir_model, model, 'cell.json'))
        param_list = [cell.get_attr(v) for k, v in param_key_dict.iteritems()]
        new_param_list = []
        for p in param_list:
            p_str = str(p)
            if 'e' in p_str:
                _, p_pow = p_str.split('e')
                order_mag = get_order_of_magnitude(p)
                if order_mag <= -10:
                    p_val = '0'
                else:
                    p_val = (('%.9f') % (p * 10**-order_mag))[:9+order_mag+1].rstrip('0') + '$\cdot 10^{-' + str(-order_mag) + '}$'
            else:
                p_val = (('%.9f') % p).rstrip('0')
                if p_val[-1] == '.':
                    p_val = p_val[:-1]
            new_param_list.append(p_val)

            # if np.abs(p) < 0.001 and p != 0:
            #     order_mag = get_order_of_magnitude(p)
            #     new_param_list.append(str(p*10**(-order_mag))+'$ 10^%i$' % -order_mag)
            # else:
            #     new_param_list.append(str(p))
        table_body[:, model_idx] = new_param_list

    table_body = np.column_stack((index, table_body))
    header = r'Parameter & Model 1 & Model 2 & Model 3 & Model 4 & Model 5 \\'
    line_break = r'\hline'
    table = tabulate(table_body, tablefmt='latex_raw')

    table_lines = table.split('\n')
    table_lines.insert(1, line_break)
    table_lines.insert(2, header)
    table = reduce(lambda a, b: a + '\n' + b, table_lines)
    print table