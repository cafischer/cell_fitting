import numpy as np
from scipy.optimize import newton
from scipy.integrate import ode
import odespy
from ode_solver import solve_implicit_euler
__author__ = 'caro'


class HHSolver:

    def __init__(self):
        pass

    def solve_impliciteuler(self, cell, t, v0, i_inj, p_gates0=None):
        dt = t[1] - t[0]
        p_gates = np.zeros(len(cell.ionchannels), dtype=object)
        current_channels = np.zeros((len(cell.ionchannels), len(t)))

        v = np.zeros(len(t))
        v[0] = v0
        for i, channel in enumerate(cell.ionchannels):
            p_gates[i] = np.zeros((channel.n_gates, len(t)))
            p_gates[i][:, 0] = channel.init_gates(v[0], p_gates0)

        for k in range(1, len(t)):
            for i, channel in enumerate(cell.ionchannels):
                for j in range(channel.n_gates):

                    def f(x_now, t_now):
                        return channel.derivative_gates(v[(t_now/dt)-1], x_now, j)

                    p_gates[i][j, k] = solve_implicit_euler(p_gates[i][j, k-1], [t[k-1], t[k]], f,
                                                            lambda x: 0)[-1]

                current_channels[i, k] = channel.compute_current(v[k-1], p_gates[i][:, k])

            def f_v(x_now, t_now):
                return cell.derivative_v(current_channels[:, k-1], i_inj[k])

            v[k] = solve_implicit_euler(v[k-1], [t[k-1], t[k]], f_v, lambda x: 0)[-1]
        return v, current_channels, p_gates


    def solve_adaptive(self, cell, t, v0, p_gates0=None):  # TODO: handle leak channel

        g = np.zeros(len(cell.ionchannels))
        p = np.zeros(len(cell.ionchannels))
        q = np.zeros(len(cell.ionchannels))
        ep = np.zeros(len(cell.ionchannels))
        for i in range(len(cell.ionchannels)):
            g[i] = cell.ionchannels[i].g_max
            p[i] = cell.ionchannels[i].power_gates[0]
            q[i] = cell.ionchannels[i].power_gates[1]
            ep[i] = cell.ionchannels[i].ep

        def f(x, ts):
            v = x[0]
            a = x[1:len(cell.ionchannels)+1]
            b = x[len(cell.ionchannels)+1:2*(len(cell.ionchannels))+1]
            i_ion = np.array([cell.ionchannels[i].compute_current(v, [a[i], b[i]])
                                        for i in range(len(cell.ionchannels))])
            dvdt = cell.derivative_v(i_ion, cell.i_inj(ts))
            dadt = [(cell.ionchannels[i].inf_gates(v)[0] - a[i]) / cell.ionchannels[i].tau_gates(v)[0]
                for i in range(len(cell.ionchannels))]
            dbdt = [(cell.ionchannels[i].inf_gates(v)[1] - b[i]) / cell.ionchannels[i].tau_gates(v)[1]
                for i in range(len(cell.ionchannels))]
            return [dvdt] + dadt + dbdt

        # initial conditions
        x0 = [v0] + [cell.ionchannels[i].init_gates(v0, p_gates0)[0] for i in range(len(cell.ionchannels))] \
             + [cell.ionchannels[i].init_gates(v0, p_gates0)[1] for i in range(len(cell.ionchannels))]

        # solve the system of odes
        solver = odespy.Vode(f, rtol=0.0, atol=1e-6, adams_or_bdf='bdf')  # TODO
        solver.set_initial_condition(x0)
        sol, t = solver.solve(t)

        v_sol = sol[:, 0]
        a_sol = sol[:, 1:len(cell.ionchannels)+1]
        b_sol = sol[:, len(cell.ionchannels)+1:2*(len(cell.ionchannels))+1]
        return v_sol, a_sol, b_sol

    def solve_gates(self, cell, t, v, p_gates0=None):
        if self.solver == 'ImplicitEuler':
            dt = t[1] - t[0]
            p_gates = np.zeros(len(cell.ionchannels), dtype=object)
            current_channels = np.zeros((len(cell.ionchannels), len(t)))

            for i, channel in enumerate(cell.ionchannels):
                inf_gates = np.zeros((channel.n_gates, len(t)))  # TODO
                tau_gates = np.zeros((channel.n_gates, len(t)))

                p_gates[i] = np.zeros((channel.n_gates, len(t)))
                p_gates0 = channel.init_gates(v[0], p_gates0)
                for j in range(channel.n_gates):
                    inf_gates[j, :] = np.array([channel.inf_gates(v[ts_i])[j] for ts_i in range(len(t))])  # TODO
                    tau_gates[j, :] = np.array([channel.tau_gates(v[ts_i])[j] for ts_i in range(len(t))])

                    def f(x_now, t_now):
                        return channel.derivative_gates(v[t_now//dt], x_now, j)

                    p_gates[i][j, :] = solve_implicit_euler(p_gates0[j], t, f, lambda x: 0)

                #current_channels[i, :] = np.array([channel.compute_current(v[k-1], p_gates[i][:, k])
                #                                   for k in range(len(t))])
                for k in range(len(t)):
                    current_channels[i, k] = channel.compute_current(v[k], p_gates[i][:, k])

            return current_channels, p_gates, inf_gates, tau_gates

    def solve_adaptive_y(self, cell, t, v_star, v0, y0, theta_idx, p_gates0=None):

        g = np.zeros(len(cell.ionchannels))
        p = np.zeros(len(cell.ionchannels))
        q = np.zeros(len(cell.ionchannels))
        ep = np.zeros(len(cell.ionchannels))
        for i in range(len(cell.ionchannels)):
            g[i] = cell.ionchannels[i].g_max
            p[i] = cell.ionchannels[i].power_gates[0]
            q[i] = cell.ionchannels[i].power_gates[1]
            ep[i] = cell.ionchannels[i].ep

        def f(x, ts):
            v_star_ts = v_star(ts)
            v = x[0]
            a = x[1:len(cell.ionchannels)+1]
            b = x[len(cell.ionchannels)+1:2*(len(cell.ionchannels))+1]
            y = x[-1]
            i_ion = np.array([cell.ionchannels[i].compute_current(v, [a[i], b[i]])
                                        for i in range(len(cell.ionchannels))])
            dvdt = cell.derivative_v(i_ion, cell.i_inj(ts))
            dadt = [(cell.ionchannels[i].inf_gates(v_star_ts)[0] - a[i]) / cell.ionchannels[i].tau_gates(v_star_ts)[0]
                for i in range(len(cell.ionchannels))]
            dbdt = [(cell.ionchannels[i].inf_gates(v_star_ts)[1] - b[i]) / cell.ionchannels[i].tau_gates(v_star_ts)[1]
                for i in range(len(cell.ionchannels))]

            dvdtdv = -1 * np.sum((g * (a.T**p) * (b.T**q)).T / cell.cm, 0)  # transpose for right application of powers
            dvdtdtheta = -1 * a[theta_idx]**p[theta_idx] * b[theta_idx]**q[theta_idx]*(v-ep[theta_idx]) / cell.cm
            dydt = 1000 * (dvdtdv * y + dvdtdtheta)
            return [dvdt] + dadt + dbdt + [dydt]

        # initial conditions
        x0 = [v0] + [cell.ionchannels[i].init_gates(v0, p_gates0)[0] for i in range(len(cell.ionchannels))] \
             + [cell.ionchannels[i].init_gates(v0, p_gates0)[1] for i in range(len(cell.ionchannels))] \
             + [y0]  # initial conditions

        # solve the system of odes
        solver = odespy.Vode(f, rtol=0.0, atol=1e-3, adams_or_bdf='bdf')
        solver.set_initial_condition(x0)
        sol, t = solver.solve(t)

        v_sol = sol[:, 0]
        a_sol = sol[:, 1:len(cell.ionchannels)+1]
        b_sol = sol[:, len(cell.ionchannels)+1:2*(len(cell.ionchannels))+2]
        y_sol = sol[:, -1]
        return v_sol, a_sol, b_sol, y_sol

    def solve_adaptive_yg(self, cell, cell_star, t, v_star0, y0, theta_idx, p_gates0=None):

        g = np.zeros(len(cell.ionchannels))
        p = np.zeros(len(cell.ionchannels))
        q = np.zeros(len(cell.ionchannels))
        ep = np.zeros(len(cell.ionchannels))
        for i in range(len(cell.ionchannels)):
            g[i] = cell.ionchannels[i].g_max
            p[i] = cell.ionchannels[i].power_gates[0]
            q[i] = cell.ionchannels[i].power_gates[1]
            ep[i] = cell.ionchannels[i].ep

        def f(x, ts):
            v_star = x[0]
            a_star = x[1:len(cell.ionchannels)+1]
            b_star = x[len(cell.ionchannels)+1:2*(len(cell.ionchannels))+1]
            v = x[2*(len(cell.ionchannels))+1]
            y = x[-1]

            i_ion_star = np.array([cell_star.ionchannels[i].compute_current(v_star, [a_star[i], b_star[i]])
                                        for i in range(len(cell.ionchannels))])
            i_ion = np.array([cell.ionchannels[i].compute_current(v, [a[i], b[i]])
                                        for i in range(len(cell.ionchannels))])

            dvdt_star = cell.derivative_v(i_ion_star, cell.i_inj(ts))
            dadt_star = [(cell.ionchannels[i].inf_gates(v_star)[0] - a_star[i]) / cell.ionchannels[i].tau_gates(v_star)[0]
                for i in range(len(cell.ionchannels))]
            dbdt_star = [(cell.ionchannels[i].inf_gates(v_star)[1] - b_star[i]) / cell.ionchannels[i].tau_gates(v_star)[1]
                for i in range(len(cell.ionchannels))]
            dvdt = cell.derivative_v(i_ion, cell.i_inj(ts))

            dvdtdv = -1 * np.sum((g * (a_star.T**p) * (b_star.T**q)).T / cell.cm, 0)  # transpose for right application of powers
            dvdtdtheta = -1 * a_star[theta_idx]**p[theta_idx] * b_star[theta_idx]**q[theta_idx]*(v-ep[theta_idx]) / cell.cm
            dydt = 1000 * (dvdtdv * y + dvdtdtheta)
            return [dvdt_star] + [dvdt] + dadt_star + dbdt_star + [dydt]

        # initial conditions
        x0 = [v_star0] \
             + [cell.ionchannels[i].init_gates(v_star0, p_gates0)[0] for i in range(len(cell.ionchannels))] \
             + [cell.ionchannels[i].init_gates(v_star0, p_gates0)[1] for i in range(len(cell.ionchannels))] \
             + [v_star0] \
             + [cell.ionchannels[i].init_gates(v_star0, p_gates0)[0] for i in range(len(cell.ionchannels))] \
             + [cell.ionchannels[i].init_gates(v_star0, p_gates0)[1] for i in range(len(cell.ionchannels))] \
             + [y0]  # initial conditions

        # solve the system of odes
        solver = odespy.Vode(f, rtol=0.0, atol=1e-3, adams_or_bdf='bdf')
        solver.set_initial_condition(x0)
        sol, t = solver.solve(t)

        v_sol = sol[:, 0]
        a_sol = sol[:, 1:len(cell.ionchannels)+1]
        b_sol = sol[:, len(cell.ionchannels)+1:2*(len(cell.ionchannels))+2]
        y_sol = sol[:, -1]
        return v_sol, a_sol, b_sol, y_sol