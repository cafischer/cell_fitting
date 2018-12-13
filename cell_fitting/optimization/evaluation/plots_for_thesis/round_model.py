import os
import numpy as np
import json
from nrn_wrapper import Cell
import re


if __name__ == '__main__':
    save_dir_model = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    model = '6'

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

        'Nap m $V_h$': ['soma', '0.5', 'nap', 'm_vh'],
        'Nap h $V_h$': ['soma', '0.5', 'nap', 'h_vh'],
        'Nat m $V_h$': ['soma', '0.5', 'nat', 'm_vh'],
        'Nat h $V_h$': ['soma', '0.5', 'nat', 'h_vh'],
        'Kdr m $V_h$': ['soma', '0.5', 'kdr', 'n_vh'],
        'HCN h $V_h$': ['soma', '0.5', 'hcn_slow', 'n_vh'],

        'Nap m $V_s$': ['soma', '0.5', 'nap', 'm_vs'],
        'Nap h $V_s$': ['soma', '0.5', 'nap', 'h_vs'],
        'Nat m $V_s$': ['soma', '0.5', 'nat', 'm_vs'],
        'Nat h $V_s$': ['soma', '0.5', 'nat', 'h_vs'],
        'Kdr m $V_s$': ['soma', '0.5', 'kdr', 'n_vs'],
        'HCN h $V_s$': ['soma', '0.5', 'hcn_slow', 'n_vs'],

        'Nap m $\\tau_{min}$': ['soma', '0.5', 'nap', 'm_tau_min'],
        'Nap h $\\tau_{min}$': ['soma', '0.5', 'nap', 'h_tau_min'],
        'Nat m $\\tau_{min}$': ['soma', '0.5', 'nat', 'm_tau_min'],
        'Nat h $\\tau_{min}$': ['soma', '0.5', 'nat', 'h_tau_min'],
        'Kdr m $\\tau_{min}$': ['soma', '0.5', 'kdr', 'n_tau_min'],
        'HCN h $\\tau_{min}$': ['soma', '0.5', 'hcn_slow', 'n_tau_min'],

        'Nap m $\\tau_{max}$': ['soma', '0.5', 'nap', 'm_tau_max'],
        'Nap h $\\tau_{max}$': ['soma', '0.5', 'nap', 'h_tau_max'],
        'Nat m $\\tau_{max}$': ['soma', '0.5', 'nat', 'm_tau_max'],
        'Nat h $\\tau_{max}$': ['soma', '0.5', 'nat', 'h_tau_max'],
        'Kdr m $\\tau_{max}$': ['soma', '0.5', 'kdr', 'n_tau_max'],
        'HCN h $\\tau_{max}$': ['soma', '0.5', 'hcn_slow', 'n_tau_max'],

        'Nap m $\\tau_{delta}$': ['soma', '0.5', 'nap', 'm_tau_delta'],
        'Nap h $\\tau_{delta}$': ['soma', '0.5', 'nap', 'h_tau_delta'],
        'Nat m $\\tau_{delta}$': ['soma', '0.5', 'nat', 'm_tau_delta'],
        'Nat h $\\tau_{delta}$': ['soma', '0.5', 'nat', 'h_tau_delta'],
        'Kdr m $\\tau_{delta}$': ['soma', '0.5', 'kdr', 'n_tau_delta'],
        'HCN h $\\tau_{delta}$': ['soma', '0.5', 'hcn_slow', 'n_tau_delta'],
    }
    channels = ['Nat', 'Nap', 'Kdr', 'HCN']
    gates = ['m', 'h']
    params = ['$g_{max}$', '$V_h$', '$V_s$', '$\\tau_{min}$', '$\\tau_{max}$', '$\\tau_{delta}$']

    # param_format = {
    #     '$c_m$': '%.2f',
    #     '$length$': '%.2f',
    #     '$diameter$': '%.2f',
    #     '$E_{Leak}$': '%.2f',
    #     '$E_{HCN}$': '%.2f',
    #     '$E_{Na}$': '%.2f',
    #     '$E_{K}$': '%.2f',
    #     '$g_{Leak}$': '%.5f',
    #     '$g_{max}$': '%.5f',
    #     '$V_h$': '%.2f',
    #     '$V_s$': '%.2f',
    #     '$\\tau_{min}$': '%.3f',
    #     '$\\tau_{max}$': '%.3f',
    #     '$\\tau_{delta}$': '%.3f',
    # }
    # param_format = {
    #     '$c_m$': '%.10f',
    #     '$length$': '%.10f',
    #     '$diameter$': '%.10f',
    #     '$E_{Leak}$': '%.10f',
    #     '$E_{HCN}$': '%.10f',
    #     '$E_{Na}$': '%.10f',
    #     '$E_{K}$': '%.10f',
    #     '$g_{Leak}$': '%.10f',
    #     '$g_{max}$': '%.10f',
    #     '$V_h$': '%.10f',
    #     '$V_s$': '%.10f',
    #     '$\\tau_{min}$': '%.10f',
    #     '$\\tau_{max}$': '%.10f',
    #     '$\\tau_{delta}$': '%.10f',
    # }
    param_format = {
        '$c_m$': '%.9f',
        '$length$': '%.9f',
        '$diameter$': '%.9f',
        '$E_{Leak}$': '%.9f',
        '$E_{HCN}$': '%.9f',
        '$E_{Na}$': '%.9f',
        '$E_{K}$': '%.9f',
        '$g_{Leak}$': '%.9f',
        '$g_{max}$': '%.9f',
        '$V_h$': '%.9f',
        '$V_s$': '%.9f',
        '$\\tau_{min}$': '%.9f',
        '$\\tau_{max}$': '%.9f',
        '$\\tau_{delta}$': '%.9f',
    }

    param_val_dict = {k: cell.get_attr(v) for k, v in param_key_dict.iteritems()}

    for param_name, param_path in param_key_dict.iteritems():
        param_val = cell.get_attr(param_path)
        split_name = param_name.split(' ')
        if len(split_name) == 2:
            param_name = split_name[1]
        elif len(split_name) == 3:
            param_name = split_name[2]
        round_by = param_format[param_name]
        round_by = re.findall(r'\d+', round_by)
        round_by = int(round_by[0])
        cell.update_attr(param_path, np.round(param_val, round_by))

    with open(os.path.join(save_dir_model, model, 'cell_rounded.json'), 'w') as f:
        json.dump(cell.get_dict(), f, indent=4)
