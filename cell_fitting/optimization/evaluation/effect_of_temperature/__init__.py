from cell_fitting.optimization.helpers import get_channel_list


def set_q10(cell, q10):
    for channel in get_channel_list(cell, 'soma'):
        if not channel == 'pas':
            cell.update_attr(['soma', '0.5', channel, 'q10'], q10)


def set_q10_g(cell, q10):
    for channel in get_channel_list(cell, 'soma'):
        if not channel == 'pas':
            cell.update_attr(['soma', '0.5', channel, 'q10_g'], q10)