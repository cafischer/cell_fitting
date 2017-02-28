import numpy as np

__author__ = 'caro'


def get_lowerbound_upperbound_keys(variables):
    lower_bound = np.zeros(len(variables))
    upper_bound = np.zeros(len(variables))
    variable_keys = list()
    for i, var in enumerate(variables):
        lower_bound[i] = var[0]
        upper_bound[i] = var[1]
        variable_keys.append(var[2])
    return lower_bound, upper_bound, variable_keys


def get_channel_list(cell, sec_name):
    mechanism_dict = cell.get_dict()[sec_name]['mechanisms']
    channel_list = [mech for mech in mechanism_dict if not '_ion' in mech]
    return channel_list


def get_ionlist(channel_list):
    """
    Get the ion names by the convention that the ion name is included in the channel name.
    :param channel_list: List of channel names.
    :type channel_list: list
    :return: List of ion names.
    :rtype: list
    """
    ion_list = []
    for channel in channel_list:
        if 'na' in channel:
            ion_list.append('na')
        elif 'k' in channel:
            ion_list.append('k')
        elif 'ca' in channel:
            ion_list.append('ca')
        else:
            ion_list.append('')
    return ion_list


def convert_unit_prefix(from_prefix, x):
    """
    Converts x from unit prefix to base unit.
    :param from_prefix: Prefix (implemented are 'da', 'd', 'c', 'm', 'u', 'n').
    :type from_prefix:str
    :param x: Quantity to convert.
    :type x: array_like
    :return: Converted quantity.
    :rtype: array_like
    """
    if from_prefix == 'T':
        return x * 1e12
    elif from_prefix == 'M':
        return x * 1e6
    elif from_prefix == 'h':
        return x * 1e2
    elif from_prefix == 'da':
        return x * 1e1
    elif from_prefix == 'd':
        return x * 1e-1
    elif from_prefix == 'c':
        return x * 1e-2
    elif from_prefix == 'm':
        return x * 1e-3
    elif from_prefix == 'u':
        return x * 1e-6
    elif from_prefix == 'n':
        return x * 1e-9
    elif from_prefix == 'p':
        return x * 1e-12
    else:
        raise ValueError('No valid prefix!')


def get_cellarea(L, diam):
    """
    Takes length and diameter of some cell segment and returns the area of that segment (assuming it to be the surface
    of a cylinder without the circle surfaces as in Neuron).
    :param L: Length (um).
    :type L: float
    :param diam: Diameter (um).
    :type diam: float
    :return: Cell area (cm).
    :rtype: float
    """
    return L * diam * np.pi