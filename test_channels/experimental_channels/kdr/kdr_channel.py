import numpy as np
from test_channels.channel_characteristics import plot_activation_curves


if __name__ == '__main__':

    v_range = np.arange(-150, 30, 0.1)

    # activation: Eder
    vh_act = -4.8
    k_act = -11
    vh_std_act = 0
    k_std_act = 0

    # inactivation: Eder
    # Inactivation is always above 0.5 -> cannot be modelled well by Boltzmann between 0 and 1!!! Very different to paper
    vh_inact = -74.3
    k_inact = 15
    vh_std_inact = 0
    k_std_inact = 0

    plot_activation_curves(v_range, vh_act, k_act, vh_std_act, k_std_act, vh_inact, k_inact, vh_std_inact, k_std_inact)