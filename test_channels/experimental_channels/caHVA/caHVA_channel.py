import numpy as np
from test_channels.channel_characteristics import plot_activation_curves


if __name__ == '__main__':

    v_range = np.arange(-150, 50, 0.1)

    # activation: Castelli and Magistretti (2006)
    vh_act = -11.1
    k_act = -8.4
    vh_std_act = 0.7
    k_std_act = 0.2

    # inactivation: Bruehl and Wadman (1999)
    vh_inact = -37
    k_inact = 9
    vh_std_inact = 3
    k_std_inact = 0.4

    plot_activation_curves(v_range, vh_act, k_act, vh_std_act, k_std_act, vh_inact, k_inact, vh_std_inact, k_std_inact)