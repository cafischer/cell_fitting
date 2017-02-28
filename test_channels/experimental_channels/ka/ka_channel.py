import numpy as np
from test_channels.channel_characteristics import plot_activation_curves

if __name__ == '__main__':

    v_range = np.arange(-150, 30, 0.1)

    # activation: Eder
    vh_act = -34.7
    k_act = -12
    vh_std_act = 0
    k_std_act = 0

    # inactivation: Eder
    vh_inact = -80.3
    k_inact = 12
    vh_std_inact = 0
    k_std_inact = 0

    plot_activation_curves(v_range, vh_act, k_act, vh_std_act, k_std_act, vh_inact, k_inact, vh_std_inact, k_std_inact)