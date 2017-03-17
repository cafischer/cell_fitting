from __future__ import division
from nrn_wrapper import *
from test_channels.test_ionchannel import *
import os
import pandas as pd


if __name__ == "__main__":

    # channel to investigate
    channel = "nap"
    model_dir = '../../../model/cells/dapmodel_nocurrents.json'
    mechanism_dir = './mod/'

    # parameters
    celsius = 24
    amps = [0, 0, 0]
    durs = [0, 480, 0]
    v_steps = np.arange(-60, -34, 5)
    stepamp = 2
    pos = 0.5
    dt = 1

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    cell.insert_mechanisms([[['soma', '0.5', channel, 'gbar']]])
    cell.update_attr(['soma', '0.5', channel, 'gbar'], 1.0)
    sec_channel = getattr(cell.soma(.5), channel)

    # change params
    vh_m = -3.0e+01
    k_m = -4.74273046
    alpha_a_h = -1.0e-04
    alpha_b_h = 1.13325424e-03
    alpha_k_h = 5.65125565
    beta_a_h = 9.97191817e-02
    beta_b_h = -2.23707028e-01
    beta_k_h = -8.29285736

    cell.soma(.5).nap.vhm = vh_m
    cell.soma(.5).nap.km = k_m
    cell.soma(.5).nap.a_alphah = alpha_a_h
    cell.soma(.5).nap.b_alphah = alpha_b_h
    cell.soma(.5).nap.k_alphah = alpha_k_h
    cell.soma(.5).nap.a_betah = beta_a_h
    cell.soma(.5).nap.b_betah = beta_b_h
    cell.soma(.5).nap.k_betah = beta_k_h
    sec_channel = getattr(cell.soma(.5), channel)

    v = np.arange(-100, 50, 1)
    from test_channels.channel_characteristics import *
    alpha = rate_constant(v, alpha_a_h, alpha_b_h, alpha_k_h)
    beta = rate_constant(v, beta_a_h, beta_b_h, beta_k_h)
    h_inf = alpha / (alpha+beta)
    h_inf_sig = boltzmann_fun(v, -53, 8)
    pl.figure()
    pl.plot(v, h_inf, 'k')
    pl.plot(v, h_inf_sig, 'r')
    pl.show()

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, celsius, amps, durs, v_steps, stepamp, pos, dt)
    #for i in range(len(i_steps)):
    #    i_steps[i] /= 20
    #plot_i_steps(i_steps, v_steps, t)

    # compare to experimental data
    all_traces = pd.read_csv(os.path.join('.', 'plots', 'digitized_vsteps', 'traces.csv'), index_col=0)
    all_traces /= np.max(np.max(np.abs(all_traces)))

    scale_fac = 1.0 / np.max(np.abs(np.matrix(i_steps)[:, 1:]))
    pl.figure()
    for i, column in enumerate(all_traces.columns):
        pl.plot(all_traces.index, all_traces[column], 'k', label=column)
        pl.plot(t[:-1], i_steps[i][1:] * scale_fac, 'r')
    pl.ylabel('Current (pA)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()