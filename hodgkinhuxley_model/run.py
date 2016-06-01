import numpy as np
import matplotlib.pyplot as pl
from mechanisms import IonChannel
from cell import Cell
from hh_solver import HHSolver

__author__ = 'caro'

if __name__ == '__main__':

    # parameter ionchannel
    g_max = 1
    ep = 80
    n_gates = 2
    power_gates = [3, 1]

    def inf_gates(v):
        return np.array([1/(1+np.exp((v-20)/5)), 1/(1+np.exp((-v+50)/5))])

    def tau_gates(v):
        return np.array([1, 1])

    # create ion channel
    ionchannel = IonChannel(g_max, ep, n_gates, power_gates, inf_gates, tau_gates)

    # parameter leak ionchannel
    g_max = 0.1
    ep = -65
    n_gates = 0
    power_gates = []

    def inf_gates(v):
        return []

    def tau_gates(v):
        return []

    # create ion channel
    ionchannel_leak = IonChannel(g_max, ep, n_gates, power_gates, inf_gates, tau_gates)

    # parameter cell
    cm = 1
    length = 16
    diam = 8
    ionchannels = [ionchannel]
    def i_inj(ts):
        if 0.02 <= ts <= 11:
            return 1
        else:
            return 0

    # create cell
    cell = Cell(cm, length, diam, ionchannels, i_inj)

    # create odesolver
    tstop = 100
    dt = 0.01
    t = np.arange(0, tstop+dt, dt)
    v0 = -65
    i_inj = np.zeros(len(t))
    i_inj[0.02/dt:11/dt] = 1
    hhsolver = HHSolver('ImplicitEuler')
    v, current, p_gates = hhsolver.solve(cell, t, v0, i_inj)
    v_adaptive, a_gate, b_gate = hhsolver.solve_adaptive(cell, t, v0)
    #current, p_gates, inf_gates, tau_gates = odesolver.solve_onlygates2(cell, t, v)

    save_dir = './test_data/'
    #with open(save_dir+'v.npy', 'r') as f:
    #    v_test = np.load(f)

    #with open(save_dir+'current_channel.npy', 'r') as f:
    #    current_test = np.load(f)

    pl.figure()
    #pl.plot(t, v_test, 'k', label='NEURON')
    pl.plot(t, v, 'b', label='Python')
    pl.plot(t, v_adaptive, 'g', label='Python adaptive')
    pl.legend()
    pl.show()

    pl.figure()
    pl.plot(t, current_test[0], 'k', label='NEURON' )
    pl.plot(t, current[0], 'b', label='Python')
    pl.legend()
    pl.show()

    pl.figure()
    pl.plot(t, current_test[1], 'k', label='NEURON' )
    pl.plot(t, current[1], 'b', label='Python')
    pl.legend()
    pl.show()

    """
    with open('./minf.npy', 'r') as f:
        minf = np.load(f)
    with open('./hinf.npy', 'r') as f:
        hinf = np.load(f)
    pl.figure()
    pl.plot(t, minf, 'k')
    pl.plot(t, inf_gates[0, :], 'b')
    pl.plot(t, hinf, 'k')
    pl.plot(t, inf_gates[1, :], 'g')
    pl.show()
    
    with open('./mtau.npy', 'r') as f:
        mtau = np.load(f)
    with open('./htau.npy', 'r') as f:
        htau = np.load(f)
    pl.figure()
    pl.plot(t, mtau, 'k')
    pl.plot(t, tau_gates[0, :], 'b')
    pl.plot(t, htau, 'k')
    pl.plot(t, tau_gates[1, :], 'g')
    pl.show()
    """