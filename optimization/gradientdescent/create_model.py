import numpy as np
import csv
import os
import pandas as pd
from fit_currents.error_analysis.model_generator import change_dt
from optimization.optimizer import extract_simulation_params
from optimization.bioinspired.problem import Problem
from hodgkinhuxley_model.mechanisms import IonChannel
from hodgkinhuxley_model.cell import Cell
from hodgkinhuxley_model.hh_solver import HHSolver
from fit_currents.error_analysis.model_generator import from_protocol

__author__ = 'caro'


def save_model(save_dir, data_name, problem, candidate, dt):
    # change dt of data
    data = change_dt(dt, problem.data, 'ramp')
    simulation_params = extract_simulation_params(data)

    # create cell
    cell = problem.get_cell(candidate)

    # run simulation and compute the variable to fit
    var_to_fit, _ = problem.fun_to_fit(cell, **simulation_params)

    nans = np.zeros(len(var_to_fit), dtype=object)
    nans[:] = np.nan
    data_model = np.column_stack((np.array(problem.data.t), np.array(problem.data.i), var_to_fit, nans, nans))
    data_model[0, 3] = 'soma'
    data_model[0, 4] = candidate

    if not os.path.exists(save_dir):
                os.makedirs(save_dir)
    with open(save_dir+data_name, 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        writer.writerow(['t', 'i', 'v', 'sec', 'candidate'])
        writer.writerows(data_model)

def save(save_dir, data_name, v, t, i, section, candidate):

    nans = np.zeros(len(v), dtype=object)
    nans[:] = np.nan
    data_model = np.column_stack((t, i, v, nans, nans))
    data_model[0, 3] = section
    data_model[0, 4] = candidate

    if not os.path.exists(save_dir):
                os.makedirs(save_dir)
    with open(save_dir+data_name, 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        writer.writerow(['t', 'i', 'v', 'sec', 'candidate'])
        writer.writerows(data_model)

def get_model():
    # make naf ionchannel
    g_max = 0.06
    ep = 80
    n_gates = 2
    power_gates = [3, 1]

    vshift = -3.5
    gbar = 0.0  # (S/cm2)

    mv = 38
    ms = 10
    mvtau = 30
    mctau = 0.14
    motau = 0.025

    hv = 62.9
    hs = 8
    hvtau = 37
    hstau = 15
    hctau = 1.15
    hotau = 0.15

    def inf_gates(v):
        minf = 1/(1 + np.exp((-(v + vshift) - mv) / ms))
        hinf = 1/(1 + np.exp(((v + vshift) + hv) / hs))
        return np.array([minf, hinf])

    def tau_gates(v):
        if(v + vshift) < -30.0:
            mtau = motau + mctau * np.exp(((v + vshift) + mvtau) / ms)
        else:
            mtau = motau + mctau * np.exp((-(v + vshift) - mvtau) / ms)
        htau = hotau + hctau / (1 + np.exp(((v + vshift) + hvtau) / hstau))
        return np.array([mtau, htau])

    naf = IonChannel(g_max, ep, n_gates, power_gates, inf_gates, tau_gates)

    # make ka ionchannel
    g_max = 0.07
    ep = -80
    n_gates = 2
    power_gates = [1, 1]

    celsius = 35
    vhalfn = 11  # (mV)
    a0n = 0.05  # (/ms)
    zetan = -1.5  # (1)
    gmn = 0.55  # (1)
    pw = -1  # (1)
    tq = -40  # (mV)
    qq = 5  # (mV)
    nmin = 0.1  # (ms)
    nscale = 1
    vhalfl = -56  # (mV)
    a0l = 0.05  # (/ms)
    zetal = 3  # (1)
    lmin = 2  # (ms)
    lscale = 1
    q10_act = 5
    q10_inact = 1
    temp = 24  # (degC)

    def alpn(v, zetan, pw, tq, qq, vhalfn, celsius):
        zeta=zetan+pw/(1+np.exp((v-tq)/qq))
        return np.exp(zeta*(v-vhalfn)*1.e-3*9.648e4/(8.315*(273.16+celsius)))

    def betn(v, zetan, pw, tq, qq, gmn, vhalfn, celsius):
        zeta=zetan+pw/(1+np.exp((v-tq)/qq))
        return np.exp(zeta*gmn*(v-vhalfn)*1.e-3*9.648e4/(8.315*(273.16+celsius)))

    def alpl(v, zetal, vhalfl, celsius):
        return np.exp(zetal*(v-vhalfl)*1.e-3*9.648e4/(8.315*(273.16+celsius)))

    def inf_gates(v):
        a = alpn(v, zetan, pw, tq, qq, vhalfn, celsius)
        ninf = 1/(1 + a)
        a = alpl(v, zetal, vhalfl, celsius)
        linf = 1/(1 + a)
        return np.array([ninf, linf])

    def tau_gates(v):
        tadj_act = q10_act**((celsius-temp)/10)
        tadj_inact = q10_inact**((celsius-temp)/10)

        a = alpn(v, zetan, pw, tq, qq, vhalfn, celsius)
        b = betn(v, zetan, pw, tq, qq, gmn, vhalfn, celsius)
        taun = (b/(a0n*(1+a))) / tadj_act
        if taun<nmin:
            taun=nmin
        taun=taun/nscale

        taul = (0.26*(v+50)) / tadj_inact
        if taul<lmin:
            taul=lmin
        taul=taul/lscale
        return np.array([taun, taul])

    ka = IonChannel(g_max, ep, n_gates, power_gates, inf_gates, tau_gates)

    # create cell
    def i_inj(ts):
        times, amps, amp_types = from_protocol('ramp')
        if len(np.nonzero(ts <= np.array(times))[0]) == 0:
            idx = len(times)-1  # in case it asks for later times return as if for last section
        else:
            idx = np.nonzero(ts <= np.array(times))[0][0]
        if idx > 0:
            idx -= 1
        if amp_types[idx] == 'const':
            return amps[idx]
        elif amp_types[idx] == 'rampup' or amp_types[idx] == 'rampdown':
            return (amps[idx+1]-amps[idx]) / (times[idx+1] - times[idx]) * (ts-times[idx]) + amps[idx]
        else:
            return None

    cm = 1
    length = 16
    diam = 8
    ionchannels = [naf, ka]
    cell = Cell(cm, length, diam, ionchannels, i_inj)
    return cell



if __name__ == '__main__':

    save_dir = './testdata/'
    data_name = 'modeldata_nafka2.csv'
    data_dir = '../bioinspired/performance_test/testdata/modeldata.csv'
    candidate = [0.06, 0.07]
    dt = 0.01

    """
    path_variables = [[['soma', 'mechanisms', 'naf', 'gbar']],
                      [['soma', 'mechanisms', 'ka', 'gbar']]]

    params = {'data_dir': data_dir,
              'model_dir': '../../model/cells/dapmodel.json',
              'mechanism_dir': '../../model/channels_currentfitting',
              'lower_bound': 0, 'upper_bound': 1,
              'maximize': False,
              'fun_to_fit': 'run_simulation', 'var_to_fit': 'v',
              'path_variables': path_variables,
              'errfun': 'errfun_featurebased'}

    problem = Problem(params)

    save_model(save_dir, data_name, problem, candidate, dt)
    """

    cell = get_model()

    data = pd.read_csv(data_dir)

    t = np.array(data.t)
    v0 = np.array(data.v)[0]
    i_inj = np.array([cell.i_inj(ts) for ts in t])


    hhsolver = HHSolver()
    v, _, _= hhsolver.solve_adaptive(cell, t, v0)

    save(save_dir, data_name, v, t, i_inj, 'soma', candidate)