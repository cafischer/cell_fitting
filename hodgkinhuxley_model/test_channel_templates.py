import numpy as np
import matplotlib.pyplot as pl
from mechanisms import IonChannel
from cell import Cell
from hh_solver import HHSolver
import pandas as pd

__author__ = 'caro'

if __name__ == '__main__':

    # parameter ionchannel
    g_max = 1
    ep = 60
    n_gates = 2
    power_gates = [3, 1]
    vm_inf = -32.5
    vh_inf = -59.8
    km_inf = -3.6
    kh_inf = 4.5
    vmtau = -32.5
    cmtau = 1
    smtau = 50
    kmtau = 4
    vhtau = -59.8
    chtau = 1
    shtau = 1
    khtau = 4

    # sx: 1-100, vx: -90-+70, kx: 10-3 -10 (fast) -10000 (slow)

    def inf_gates(v):
        return np.array([1/(1+np.exp((v-vm_inf)/km_inf)), 1/(1+np.exp((-v+vh_inf)/kh_inf))])

    def tau_gates_constant(v):
        return np.array([1, 1])

    def tau_gates_derboltz(v):
        return np.array([kmtau * np.exp(kmtau * (v-vmtau)) / (np.exp(kmtau*v) + np.exp(kmtau*vmtau)),
                         khtau * np.exp(khtau * (v-vhtau)) / (np.exp(khtau*v) + np.exp(khtau*vhtau))])

    def tau_gates_gauss(v):
        return np.array([cmtau * np.exp(-(v-vmtau)**2/(2*smtau**2)),
                         chtau * np.exp(-(v-vhtau)**2/(2*shtau**2))])

    # plot tau function
    v_range = np.arange(-90, 70, 0.1)
    pl.figure()
    pl.plot(v_range, tau_gates_gauss(v_range)[0])
    pl.plot(v_range, tau_gates_gauss(v_range)[1])
    pl.show()

    # create ion channel
    ionchannel = IonChannel(g_max, ep, n_gates, power_gates, inf_gates, tau_gates_gauss)

    # parameter cell
    cm = 1
    length = 16
    diam = 8
    ionchannels = [ionchannel]

    # create cell
    cell = Cell(cm, length, diam, ionchannels)

    # load data
    save_dir = '../data/new_cells/2015_08_11d/dap/dap.csv'
    data = pd.read_csv(save_dir)
    v = np.array(data.v)

    # create odesolver
    #tstop = 100
    #dt = 0.01
    t = np.array(data.t)  #np.arange(0, tstop+dt, dt)
    odesolver = HHSolver('ImplicitEuler')

    # compute channel current
    current, p_gates, inf_gates, tau_gates = odesolver.solve_gates(cell, t, v)

    # plots
    pl.figure()
    pl.plot(t, -1 * current[0], 'b', label='Python')
    pl.plot(t, (v-v[0]) * np.max(-1 * current[0]) / np.max(v-v[0]), 'k')
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