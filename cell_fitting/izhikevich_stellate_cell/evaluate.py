import json
from cell_fitting.izhikevich_stellate_cell import get_v_izhikevich
from cell_fitting.optimization.fitter import IzhikevichFitter
import matplotlib.pyplot as pl
from cell_fitting.optimization.errfuns import rms

save_dir = './results/L-BFGS-B/ramp/'
candidate = [4.421702827014319, 0.026529749064180107, 0.0090398616994982594, 1.1377248310015808, -6.0025700499387193]

with open(save_dir+'problem.json', 'r') as f:
    problem_specification = json.load(f)
problem = IzhikevichFitter(**problem_specification)

for name, value in zip(problem.name_variables, candidate):
    problem.name_value_variables[name] = value

for i, sim_params in enumerate(problem.simulation_params):
    i_inj = sim_params['i_inj']
    tstop = sim_params['tstop']
    dt = sim_params['dt']
    v_candidate, t_candidate, _ = get_v_izhikevich(i_inj, tstop, dt, **problem.name_value_variables)

    print rms(v_candidate, problem.data[i].v.values)

    pl.figure()
    pl.plot(problem.data[i].t, problem.data[i].v, 'k')
    pl.plot(t_candidate, v_candidate, 'r')
    pl.show()

    #pl.figure()
    #pl.plot(problem.data[i].t, problem.data[i].i, 'k')
    #pl.show()