import numpy as np
from test_channels.channel_characteristics import plot_activation_curves


if __name__ == '__main__':

    v_range = np.arange(-150, 50, 0.1)

    # activation: Magistretti
    vh_act = -32.5
    k_act = -3.6
    vh_std_act = 6.5
    k_std_act = 0.9

    # inactivation: Magistretti
    vh_inact = -59.8
    k_inact = 4.5
    vh_std_inact = 5.2
    k_std_inact = 0.9

    plot_activation_curves(v_range, vh_act, k_act, vh_std_act, k_std_act, vh_inact, k_inact, vh_std_inact, k_std_inact)