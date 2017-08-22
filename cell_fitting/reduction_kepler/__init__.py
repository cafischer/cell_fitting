import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as pl
from cell_fitting.optimization.simulate import extract_simulation_params, simulate_gates, iclamp_handling_onset
from nrn_wrapper import Cell
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == '__main__':
    # parameters
    data_dir = '../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv'
    save_dir = '../results/server/2017-07-06_13:50:52/434/L-BFGS-B/'
    model_dir = os.path.join(save_dir, 'model', 'cell.json')
    mechanism_dir = '../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # get simulation_params
    data = pd.read_csv(data_dir)
    simulation_params = extract_simulation_params(data)

    # get gates
    gates, power_gates, vh_gates, vs_gates = simulate_gates(cell, simulation_params, return_vh_vs=True)
    t = np.arange(0, simulation_params['tstop']+simulation_params['dt'], simulation_params['dt'])

    # plot equivalent potentials
    def compute_equivalent_potential(x, x_vh, x_vs):
        return - np.log(1/x - 1) * x_vs + x_vh

    equivalent_potentials = {}
    for gate in gates.keys():
        equivalent_potentials[gate] = compute_equivalent_potential(gates[gate], vh_gates[gate], vs_gates[gate])

    # plot
    # for k in gates.keys():
    #     pl.plot(t, equivalent_potentials[k], label=k)
    #     pl.ylabel('Equivalent Potential', fontsize=16)
    #     pl.xlabel('Time (ms)', fontsize=16)
    #     pl.legend(fontsize=16)
    # pl.show()

    # # find best linear fits between equivalent potentials
    # for gate in gates.keys():
    #     for other_gate in gates.keys():
    #         reg = linear_model.LinearRegression()
    #         reg.fit(np.array([gates[other_gate]]).T, np.array([gates[gate]]).T)
    #
    #         print gate+' = a * '+other_gate+ ' + b'
    #         print('Coefficients: \n', reg.coef_)
    #         print('Intercept: \n', reg.intercept_)
    #         gate_pred = reg.predict(np.array([gates[other_gate]]).T)
    #         #print("Mean squared error %.5f: "
    #         #      % mean_squared_error(gates[gate], gate_pred))
    #         print("Variance score %.2f: " % r2_score(gates[gate], gate_pred))  # 1 is perfect prediction
    #
    #         pl.figure()
    #         pl.plot(t, gates[gate], 'k')
    #         pl.plot(t, gate_pred, 'r')
    #         pl.show()
    #
    #         pl.figure()
    #         pl.plot(gates[other_gate], gates[gate], 'k')
    #         pl.plot(gates[other_gate], reg.coef_[0] * gates[other_gate] + reg.intercept_[0], 'r')
    #         pl.show()

    # replace gate
    v_b, t_b, i_inj = iclamp_handling_onset(cell, **simulation_params)

    # 'Coefficients: \n', array([[ 0.88453378]]))
    # ('Intercept: \n', array([-0.12882545]))
    slope = 0.88453378
    intercept = -0.12882545
    g_bar = cell.soma(.5).nat.gbar
    cell.soma(.5).nat.gbar = 0
    cell.soma.insert("nat_ep")
    cell.soma(.5).nat_ep.a = slope
    cell.soma(.5).nat_ep.b = intercept
    cell.soma(.5).nat_ep.gbar = g_bar
    cell.soma(.5).nat_ep.m_vh = cell.soma(.5).nat.m_vh
    cell.soma(.5).nat_ep.m_vs = cell.soma(.5).nat.m_vs
    cell.soma(.5).nat_ep.h_vh = cell.soma(.5).nat.h_vh
    cell.soma(.5).nat_ep.h_vs = cell.soma(.5).nat.h_vs
    cell.soma(.5).nat_ep.h_tau_min = cell.soma(.5).nat.h_tau_min
    cell.soma(.5).nat_ep.h_tau_max = cell.soma(.5).nat.h_tau_max
    cell.soma(.5).nat_ep.h_tau_delta = cell.soma(.5).nat.h_tau_delta
    cell.soma(.5).nat_ep.m_pow = cell.soma(.5).nat.m_pow
    cell.soma(.5).nat_ep.h_pow = cell.soma(.5).nat.h_pow

    cell.soma(.5).nat_ep.n_vh = cell.soma(.5).kdr.n_vh
    cell.soma(.5).nat_ep.n_vs = cell.soma(.5).kdr.n_vs
    cell.soma(.5).nat_ep.n_tau_min = cell.soma(.5).kdr.n_tau_min
    cell.soma(.5).nat_ep.n_tau_max = cell.soma(.5).kdr.n_tau_max
    cell.soma(.5).nat_ep.n_tau_delta = cell.soma(.5).kdr.n_tau_delta

    from neuron import h
    n = h.Vector()
    n.record(cell.soma(.5).kdr._ref_n)
    n_nat_ep = h.Vector()
    n_nat_ep.record(cell.soma(.5).nat_ep._ref_n)
    m_nat_ep = h.Vector()
    m_nat_ep.record(cell.soma(.5).nat_ep._ref_m)
    m_nat = h.Vector()
    m_nat.record(cell.soma(.5).nat._ref_m)
    h_nat_ep = h.Vector()
    h_nat_ep.record(cell.soma(.5).nat_ep._ref_h)
    h_nat = h.Vector()
    h_nat.record(cell.soma(.5).nat._ref_h)
    v_a, t_a, i_inj = iclamp_handling_onset(cell, **simulation_params)

    pl.figure()
    pl.plot(t_b, v_b, 'k')
    pl.plot(t_a, v_a, 'r')
    pl.show()

    n = np.array(n)
    n_nat_ep = np.array(n_nat_ep)
    m_nat_ep = np.array(m_nat_ep)
    h_nat_ep = np.array(h_nat_ep)
    h_nat = np.array(h_nat)

    pl.figure()
    pl.title('check n is the same')
    pl.plot(n, 'k', label='kdr')
    pl.plot(n_nat_ep, 'r', label='nat_ep')
    pl.legend()
    pl.show()

    pl.figure()
    pl.title('check h is the same')
    pl.plot(h_nat, 'k', label='nat')
    pl.plot(h_nat_ep, 'r', label='nat_ep')
    pl.legend()
    pl.show()

    pl.figure()
    pl.title('check m_new and m_old similar')
    pl.plot(m_nat, 'k', label='nat')
    pl.plot(m_nat_ep, 'r', label='nat_ep')
    pl.legend()
    pl.show()
    # TODO: verschoben

    n_vh = cell.soma(.5).kdr.n_vh
    n_vs = cell.soma(.5).kdr.n_vs
    m_vh = cell.soma(.5).nat.m_vh
    m_vs = cell.soma(.5).nat.m_vs
    h_vh = cell.soma(.5).nat.h_vh
    h_vs = cell.soma(.5).nat.h_vs
    m_vh_ep = cell.soma(.5).nat_ep.m_vh
    m_vs_ep = cell.soma(.5).nat_ep.m_vs
    h_tau_min = cell.soma(.5).nat.h_tau_min
    h_tau_max = cell.soma(.5).nat.h_tau_max
    h_tau_delta = cell.soma(.5).nat.h_tau_delta
    h_vh_ep = cell.soma(.5).nat_ep.h_vh
    h_vs_ep = cell.soma(.5).nat_ep.h_vs
    h_tau_min_ep = cell.soma(.5).nat_ep.h_tau_min
    h_tau_max_ep = cell.soma(.5).nat_ep.h_tau_max
    h_tau_delta_ep = cell.soma(.5).nat_ep.h_tau_delta
    assert m_vh == m_vh_ep
    assert m_vs == m_vs_ep
    assert h_vh == h_vh_ep
    assert h_vs == h_vs_ep
    assert h_tau_min == h_tau_min_ep
    assert h_tau_max == h_tau_max_ep
    assert h_tau_delta == h_tau_delta_ep

    pl.figure()
    pl.title('check ve_m = a * ve_n + b')
    pl.plot(1 / (1 + np.exp((slope * np.log(1/n-1) * n_vs - slope * n_vh - intercept + m_vh) / m_vs)), 'k')
    pl.plot(m_nat_ep, 'r')
    pl.show()

    # pl.figure()
    # pl.title('check ve_m = a * ve_n + b')
    # pl.plot(slope*compute_equivalent_potential(n, n_vh, n_vs) + intercept, 'k')
    # pl.plot(compute_equivalent_potential(m_nat_ep, m_vh, m_vs), 'r')
    # pl.show()