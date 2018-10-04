import numpy as np
import matplotlib


def init_nan(shape):
    x = np.empty(shape)
    x[:] = np.nan
    return x


def merge_dicts(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict


def norm(x, lower_bound, upper_bound):
    return [(x[i] - lower_bound[i]) / (upper_bound[i] - lower_bound[i]) for i in range(len(x))]


def unnorm(x, lower_bound, upper_bound):
    return [x[i] * (upper_bound[i] - lower_bound[i]) + lower_bound[i] for i in range(len(x))]


def convert_from_unit(prefix, x):
    """
    Converts x from unit prefix to base unit.
    :param prefix: Prefix (implemented are 'T', 'M', da', 'd', 'c', 'm', 'u', 'n', 'p').
    :type prefix:str
    :param x: Quantity to convert.
    :type x: array_like
    :return: Converted quantity.
    :rtype: array_like
    """
    if prefix == 'T':
        return x * 1e12
    elif prefix == 'M':
        return x * 1e6
    elif prefix == 'h':
        return x * 1e2
    elif prefix == 'da':
        return x * 1e1
    elif prefix == 'd':
        return x * 1e-1
    elif prefix == 'c':
        return x * 1e-2
    elif prefix == 'm':
        return x * 1e-3
    elif prefix == 'u':
        return x * 1e-6
    elif prefix == 'n':
        return x * 1e-9
    elif prefix == 'p':
        return x * 1e-12
    else:
        raise ValueError('No valid prefix!')


def convert_to_unit(prefix, x):
    """
    Converts x from base unit to unit prefix.
    :param prefix: Prefix (implemented are 'T', 'M', da', 'd', 'c', 'm', 'u', 'n', 'p').
    :type prefix:str
    :param x: Quantity to convert.
    :type x: array_like
    :return: Converted quantity.
    :rtype: array_like
    """
    if prefix == 'T':
        return x * 1e-12
    elif prefix == 'M':
        return x * 1e-6
    elif prefix == 'h':
        return x * 1e-2
    elif prefix == 'da':
        return x * 1e-1
    elif prefix == 'd':
        return x * 1e1
    elif prefix == 'c':
        return x * 1e2
    elif prefix == 'm':
        return x * 1e3
    elif prefix == 'u':
        return x * 1e6
    elif prefix == 'n':
        return x * 1e9
    elif prefix == 'p':
        return x * 1e12
    else:
        raise ValueError('No valid prefix!')


def change_color_brightness(color, percent, direction='brighter'):
    '''
    Makes color brighter by the given percentage.
    :param color: Color in rgb (in [0, 1]).
    :param percent: Percentage of brightening.
    :return: Brightened color in rgb (in [0, 1])
    '''
    color = np.array(color)
    if direction == 'brighter':
        comp_color = np.array(matplotlib.colors.to_rgb('w'))
    elif direction == 'darker':
        comp_color = np.array(matplotlib.colors.to_rgb('k'))
    else:
        raise ValueError('direction can only be brighter or darker!')
    return color * (1 - percent/100.) + comp_color * (percent/100.0)


def get_gates_of_channel():
    return {
        'nat': ['m', 'h'],
        'nap': ['m', 'h'],
        'hcn': ['n'],
        'kdr': ['n'],
        'pas': []
    }


def get_channel_dict_for_plotting():
    return {
        'nat': '$Na_T$',
        'nap': '$Na_P$',
        'hcn': '$HCN$',
        'kdr': '$K_{DR}$',
        'pas': 'Leak'
    }


def get_gate_dict_for_plotting():
    return {
        'nat_m': '$act.$',
        'nat_h': '$inact.$',
        'nap_m': '$act.$',
        'nap_h': '$inact.$',
        'hcn_n': '$inact.$',
        'kdr_n': '$act.$',
        'pas': ''
    }


def parameter_dict_for_plotting():
    return {
        'e': 'E',
        'ehcn': 'E',
        'gbar': '$g_{max}$',
        'g': '$g_{max}$',
        'cm': '$c_m$',
        'vh': '$V_h$',
        'vs': '$V_s$',
        'tau_min': r'$\tau_{min}$',
        'tau_max': r'$\tau_{max}$',
        'tau_delta': r'$\tau_{delta}$'
    }


def get_channel_color_for_plotting():
    return {
        'nat': 'r',
        'nap': 'b',
        'hcn': 'y',
        'kdr': 'g',
        'pas': '0.5'
    }


def characteristics_dict_for_plotting():
    return {
        'AP_amp': 'AP amp.',
        'AP_width': 'AP width',
        'DAP_amp': 'DAP amp.',
        'DAP_width': 'DAP width',
        'DAP_deflection': 'DAP deflection',
        'DAP_time': '$Time_{AP-DAP}$',
        'fAHP_amp': 'fAHP amp.',
    }


def get_variable_names_for_plotting(variable_names):
    channel_dict = get_channel_dict_for_plotting()
    gate_dict = get_gate_dict_for_plotting()
    parameter_dict = parameter_dict_for_plotting()
    new_variable_names = np.zeros(len(variable_names), dtype=object)
    for i, v in enumerate(variable_names):
        v_split = v.split(' ')
        if 'soma' in v:
            new_variable_names[i] = 'Soma ' + parameter_dict[v_split[1]]
            continue

        if v_split[0] == 'hcn_slow':
            v_split[0] = 'hcn'
        if '_' in v_split[1]:
            v_s2 = v_split[1].split('_')
            if len(v_s2) == 3:
                param = v_s2[1] + '_' + v_s2[2]
            else:
                param = v_s2[1]
            new_variable_names[i] = channel_dict[v_split[0]] + ' ' + gate_dict[v_split[0] + '_' + v_s2[0]] + ' ' + \
                                    parameter_dict[param]
        else:
            new_variable_names[i] = channel_dict[v_split[0]] + ' ' + parameter_dict[v_split[1]]
    return new_variable_names