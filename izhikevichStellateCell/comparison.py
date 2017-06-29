import pandas as pd
import matplotlib.pyplot as pl
import numpy as np

from izhikevichStellateCell import get_v_izhikevich, get_v_izhikevich_vector2d, phase_plot

step = True
ramp = True

cell = '2015_08_26b'
# 249.45730794687225 0.9191785392903827 0.01913708725391569 14.600701419091514
if step:
    cm = 249.457  # 300
    k_rest = 0.919  # 1.3
    k_t = 0.9
    a1 = 0.0191 # 0.017
    b1 = 14.601 # 27
    d1 = 0
    a2 = 0
    b2 = 0
    d2 = 0
    v_rest = -62.5
    v_t = -47.0
    v_reset = -49.0
    v_peak = 51.5
    i_b = 0

    cm = 185  # 300
    k_rest = 0.5  # 1.3
    k_t = 200
    a1 = 0.0072
    b1 = 28.21
    d1 = 0.73
    a2 = 1.026
    b2 = 2.049
    d2 = -531.63
    v_rest = -62.5
    v_t = -47.0
    v_reset = -49.0
    v_peak = 51.5
    i_b = 0
    v0 = v_rest
    u0 = [0, 0]

    #data_dir = '../data/' + cell + '/IV/-0.15(nA).csv'
    data_dir = '../data/' + cell + '/raw/IV/0.25(nA).csv'
    data = pd.read_csv(data_dir)

    # shift data to account for changing resting potential of the cell (to -63mV)
    #data.v += 0.46

    tstop = data.t.values[-1]
    dt = data.t.values[1]
    i_inj = data.i.values #* -1.5 # TODO

    #data_save = pd.DataFrame()
    #data_save['v'] = data.v[int(200/dt):int(285/dt)].values
    #data_save['t'] = data.t[int(200/dt):int(285/dt)].values - 200
    #data_save['i'] = data.i[int(200/dt):int(285/dt)].values
    #data_save.to_csv('./data/hyptester_shortened.csv', index=False)

    v_model, t, u_model = get_v_izhikevich_vector2d(i_inj, tstop, dt, v_rest, v_t, v_reset, v_peak, cm, k_rest, k_t,
                                              a1, a2, b1, b2, d1, d2, i_b, v0, u0)

    pl.figure()
    pl.plot(data.t, data.v, 'k')
    pl.plot(t, v_model, 'r')
    pl.show()

    #phase_plot(-80, 50, -60, 180, v_rest, v_t, cm, k_rest, a1, b1, i_b, v_model, u_model[0, :])


# ramp 0.007164990264486751 28.21011601603898 0.7288995370579201 1.0261751643463048 2.0485586109917415 -531.6310742985556
# 0.01 25.86767580859483 2.0 -1.210569208740523 0.0
if ramp:
    cm = 185  # 300
    k_rest = 0.75  # 1.3
    k_t = 200
    a1 = 0.0072
    b1 = 28.21
    d1 = 0.73
    a2 = 1.026
    b2 = 2.049
    d2 = -531.63
    v_rest = -62.5
    v_t = -47.0
    v_reset = -49.0
    v_peak = 51.5
    i_b = 0
    v0 = v_rest
    u0 = [0, 0]

    data_dir = '../data/2015_08_26b/raw/rampIV/3.0(nA).csv'
    data = pd.read_csv(data_dir)

    tstop = data.t.values[-1]
    dt = data.t.values[1]
    i_inj = np.array(data.i.values)

    v_model, t, u_model = get_v_izhikevich_vector2d(i_inj, tstop, dt, v_rest, v_t, v_reset, v_peak, cm, k_rest, k_t,
                                                    a1, a2, b1, b2, d1, d2, i_b, v0, u0)

    idxs = np.where(v_model == v_peak)[0]
    if len(idxs) >= 1:
        idx_start = idxs[0] + 1
        idx_end = idx_start + len(data.v.values[int(13.6/dt):int(120/dt)])

        pl.figure()
        pl.plot(data.t[int(13.6/dt):int(120/dt)], data.v[int(13.6/dt):int(120/dt)], 'k')
        pl.plot(data.t[int(13.6/dt):int(120/dt)], v_model[idx_start:idx_end], 'r')

    pl.figure()
    pl.plot(data.t, data.v, 'k')
    pl.plot(t, v_model, 'r')

    phase_plot(-80, 50, -300, 100, v_rest, v_t, cm, k_rest, a1, b1, i_b, v_model, u_model[0, :])
    phase_plot(-80, 50, -300, 100, v_rest, v_t, cm, k_rest, a2, b2, i_b, v_model, u_model[1, :])
    pl.show()