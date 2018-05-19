import matplotlib.pyplot as plt
import numpy as np


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


def phase_plot(vmin, vmax, umin, umax, v_rest, v_t, cm, k, a, b, i_b, v_trajectory=None, u_trajectory=None):

    V, U = np.meshgrid(np.arange(vmin, vmax+5, 5), np.arange(umin, umax+5, 5))
    dvdt = (k * (V - v_rest) * (V - v_t) + i_b - U) / cm
    dudt = (a * (b * (V - v_rest) - U))

    def vcline(v):
        u = k * (v - v_rest) * (v - v_t) + i_b
        return u

    def ucline(v):
        u = b * (v - v_rest)
        return u

    v_range = np.arange(vmin, vmax, 0.1)

    plt.figure()
    plt.title('Phase Plot')
    plt.quiver(V, U, dvdt, dudt, color='k', angles='xy', scale_units='xy', scale=2.5)
    if v_trajectory is not None and u_trajectory is not None:
        plt.plot(v_trajectory, u_trajectory, 'g')
    plt.plot(v_range, vcline(v_range), '-r')
    plt.plot(v_range, ucline(v_range), '-b')
    plt.xlim([vmin, vmax])
    plt.ylim([umin, umax])
    plt.show()


if __name__ == '__main__':
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
    i_inj = np.zeros(int(round(tstop / dt)))
    i_inj[int(10/dt):int(12/dt)] = 0.02

    v, t, u = get_v_izhikevich(i_inj, tstop, dt, v_rest, v_t, v_reset, v_peak, cm, k, k, a, b, d, i_b, v0, u0)

    plt.figure()
    plt.plot(t, v)

    #phase_plot(-80, 30, -30, 10, v_rest, v_t, cm, k, a, b, i_b, v, u)

    phase_plot(-80, 20, -30, 10, v_rest, v_t, cm, k, a, b, i_b, v, u)
    plt.show()
