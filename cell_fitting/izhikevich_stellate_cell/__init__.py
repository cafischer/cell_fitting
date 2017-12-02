import matplotlib.pyplot as pl
import numpy as np
from cell_characteristics import to_idx


def get_v_izhikevich(i_inj, tstop, dt, v_rest, v_t, v_reset, v_peak, cm, k_rest, k_t, a, b, d, i_b, v0, u0):

    t = np.arange(0, tstop + dt, dt)
    i_inj_pA = np.array(i_inj) * 1000
    v = np.zeros(len(t))
    u = np.zeros(len(t))
    v[0] = v0
    u[0] = u0

    # solve using forward Euler
    for i in range(0, len(t)-1):
        if v[i] <= v_t:
            v[i+1] = v[i] + dt * (k_rest * (v[i] - v_rest) * (v[i] - v_t) + i_b - u[i] + i_inj_pA[i]) / cm
        else:
            v[i+1] = v[i] + dt * (k_t * (v[i] - v_rest) * (v[i] - v_t) + i_b - u[i] + i_inj_pA[i]) / cm
        u[i+1] = u[i] + dt * (a * (b * (v[i] - v_rest) - u[i]))
        if v[i] >= v_peak:
            v[i] = v_peak
            v[i+1] = v_reset
            u[i+1] += d

    if v[-1] >= v_peak:
        v[-1] = v_peak

    return v, t, u


def get_v_izhikevich_vector2d(i_inj, tstop, dt, v_rest, v_t, v_reset, v_peak, cm, k_rest, k_t, a1, a2, b1, b2, d1, d2, i_b,
                              v0, u0):
    a = np.array([a1, a2])
    b = np.array([b1, b2])
    d = np.array([d1, d2])
    return get_v_izhikevich_vector(i_inj, tstop, dt, v_rest, v_t, v_reset, v_peak, cm, k_rest, k_t, a, b, d, i_b,
                                   v0, u0)


def get_v_izhikevich_vector(i_inj, tstop, dt, v_rest, v_t, v_reset, v_peak, cm, k_rest, k_t, a, b, d, i_b, v0, u0):

    t = np.arange(0, tstop + dt, dt)
    i_inj_pA = np.array(i_inj) * 1000
    v = np.zeros(len(t))
    u = np.zeros((len(u0), len(t)))
    v[0] = v0
    u[:, 0] = u0

    # solve using forward Euler
    for i in range(0, len(t)-1):
        if v[i] <= v_t:
            v[i+1] = v[i] + dt * (k_rest * (v[i] - v_rest) * (v[i] - v_t) + i_b - np.sum(u[:, i]) + i_inj_pA[i]) / cm
        else:
            v[i + 1] = v[i] + dt * (k_t * (v[i] - v_rest) * (v[i] - v_t) + i_b - np.sum(u[:, i]) + i_inj_pA[i]) / cm
        u[:, i+1] = u[:, i] + dt * (a * (b * (v[i] - v_rest) - u[:, i]))
        if v[i] >= v_peak:
            v[i] = v_peak
            v[i+1] = v_reset
            u[:, i+1] += d
    if v[-1] >= v_peak:
        v[-1] = v_peak
    return v, t, u


def phase_plot(vmin, vmax, umin, umax, v_rest, v_t, cm, k, a, b, i_b, v_trajectory=None, u_trajectory=None):

    V, U = np.meshgrid(np.arange(vmin, vmax, 3), np.arange(umin, umax, 15))
    dvdt = (k * (V - v_rest) * (V - v_t) + i_b - U) / cm
    dudt = (a * (b * (V - v_rest) - U))

    def vcline(v):
        u = k * (v - v_rest) * (v - v_t) + i_b
        return u

    def ucline(v):
        u = b * (v - v_rest)
        return u

    v_range = np.arange(vmin, vmax, 0.1)

    pl.figure()
    pl.title('Phase Plot')
    pl.quiver(V, U, dvdt, dudt, color='k', angles='xy', scale_units='xy', scale=0.5)
    if v_trajectory is not None and u_trajectory is not None:
        pl.plot(v_trajectory, u_trajectory, 'g')
    pl.plot(v_range, vcline(v_range), '-r')
    pl.plot(v_range, ucline(v_range), '-b')
    pl.xlim([vmin, vmax])
    pl.ylim([umin, umax])
    pl.show()


def get_v_izhikevich_coupled(i_inj, tstop, dt, v_rest, v_t, v_reset, v_peak, cm, k_rest, k_t, a, b, d, i_b, v0, u0, couple,
                        v_rest_d, v_t_d, v_reset_d, v_peak_d, cm_d, k_rest_d, k_t_d, a_d, b_d, d_d, i_b_d, v0_d, u0_d, couple_d):

    t = np.arange(0, tstop + dt, dt)
    i_inj_pA = np.array(i_inj) * 1000
    v = np.zeros(len(t))
    u = np.zeros(len(t))
    v[0] = v0
    u[0] = u0
    v_d = np.zeros(len(t))
    u_d = np.zeros(len(t))
    v_d[0] = v0_d
    u_d[0] = u0_d

    # solve using forward Euler
    for i in range(0, len(t) - 1):
        if v[i] <= v_t:
            v[i + 1] = v[i] + dt * (k_rest * (v[i] - v_rest) * (v[i] - v_t) + i_b - u[i] + i_inj_pA[i] + couple*(v_d[i]-v[i])) / cm
        else:
            v[i + 1] = v[i] + dt * (k_t * (v[i] - v_rest) * (v[i] - v_t) + i_b - u[i] + i_inj_pA[i] + couple*(v_d[i]-v[i])) / cm
        u[i + 1] = u[i] + dt * (a * (b * (v[i] - v_rest) - u[i]))
        if v[i] >= v_peak:
            v[i] = v_peak
            v[i + 1] = v_reset
            u[i + 1] += d

        if v_d[i] <= v_t_d:
            v_d[i + 1] = v_d[i] + dt * (k_rest_d * (v_d[i] - v_rest_d) * (v_d[i] - v_t_d) + i_b_d - u_d[i] + couple_d*(v[i]-v_d[i])) / cm_d
        else:
            v_d[i + 1] = v_d[i] + dt * (k_t_d * (v_d[i] - v_rest_d) * (v_d[i] - v_t_d) + i_b_d - u_d[i] + couple_d*(v[i]-v_d[i])) / cm_d
        u_d[i + 1] = u_d[i] + dt * (a_d * (b_d * (v_d[i] - v_rest_d) - u_d[i]))
        if v_d[i] >= v_peak_d:
            v_d[i] = v_peak_d
            v_d[i + 1] = v_reset_d
            u_d[i + 1] += d_d

    if v[-1] >= v_peak:
        v[-1] = v_peak
    if v_d[-1] >= v_peak_d:
        v_d[-1] = v_peak_d

    return v, t, u, v_d, u_d


if __name__ == '__main__':
    """
    # replicate DAP Model of the Izhikevich Book
    cm = 1
    k = 0.04
    v_rest = 0
    v_t = -5 / k
    v_reset = -60
    v_peak = 30
    a = 1
    b = 0.2
    d = -21
    i_b = 140
    v0 = -70
    u0 = v0 * b

    tstop = 50  # ms
    dt = 0.1  # ms
    i_inj = np.zeros(tstop/dt)
    i_inj[int(10/dt):int(12/dt)] = 0.02

    v, t, u = get_v_izhikevich(i_inj, tstop, dt, v_rest, v_t, v_reset, v_peak, cm, k, k, a, b, d, i_b, v0, u0)

    pl.figure()
    pl.plot(t, v)

    phase_plot(-80, 30, -30, 10, v_rest, v_t, cm, k, a, b, i_b, v, u)
    pl.show()

    """
    # replicate DAP from dendritic spike of the Izhikevich Book

    # soma
    cm = 150
    k = 3
    couple = 50
    v_rest = -70
    v_t = -45
    v_reset = -52
    v_peak = 50
    a = 0.01
    b = 5
    d = 240
    i_b = 0
    v0 = -64
    u0 = v0 * b

    # Active dendrite:
    cm_d = 30
    k_d = 1
    couple_d = 20
    v_rest_d = -50
    v_t_d = -50
    v_reset_d = -20
    v_peak_d = 20
    a_d = 3
    b_d = 15
    d_d = 500
    i_b_d = 0
    v0_d = -64
    u0_d = v0_d * b_d

    tstop = 800  # ms
    dt = 0.1  # ms
    i_inj = np.zeros(to_idx(tstop, dt))
    i_inj[int(400/dt):int(405/dt)] = 1.

    v, t, u, v_d, u_d = get_v_izhikevich_coupled(i_inj, tstop, dt, v_rest, v_t, v_reset, v_peak, cm, k, k, a, b, d, i_b,
                                       v0, u0, couple, v_rest_d, v_t_d, v_reset_d, v_peak_d, cm_d, k_d, k_d, a_d,
                                       b_d, d_d, i_b_d, v0_d, u0_d, couple_d)

    pl.figure()
    pl.plot(t, v, 'k')
    pl.plot(t, v_d, 'b')

    #phase_plot(-80, 30, -30, 10, v_rest, v_t, cm, k, a, b, i_b, v, u)
    pl.show()