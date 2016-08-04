import json
import re
import matplotlib.pyplot as pl

from optimization import problems
from optimization.problems import *
from optimization.simulate import run_simulation

__author__ = 'caro'


if __name__ == '__main__':

    method = 'PSO'
    save_dir = '../../results/bio_inspired/DAP_fromAP/'
    trial = 0

    with open(save_dir+'problem.json', 'r') as f:
        params = json.load(f)
    problem = getattr(problems, params['name'])(**params)

    n_params = len(problem.path_variables)

    path = save_dir+method+'/individuals_file_'+str(trial)+'.csv'
    individuals_file = pd.read_csv(path, dtype={'generation': np.int64, 'number': np.int64, 'fitness': np.float64,
                                                'candidate': str})
    n_generations = individuals_file.generation.iloc[-1]
    best = individuals_file.index[np.logical_and(individuals_file.generation.isin([n_generations]),
                                                 individuals_file.number.isin([0]))]  # 1st candidate of last generation
    candidate = individuals_file.candidate.iloc[best].values[0]
    candidate = re.sub('[\[\]]', '', candidate)  # convert string representation of candidate back to float
    candidate = np.array([float(x) for x in candidate.split()])

    data = pd.read_csv(params['data_dir'])

    # update cell wth candidate variables
    problem.update_cell(candidate)

    # record currents from the soma
    channel_list = get_channel_list(problem.cell, 'soma')
    ion_list = get_ionlist(channel_list)
    currents = [problem.cell.soma.record_from(channel_list[k], 'i'+ion_list[k], pos=.5) for k in range(len(channel_list))]

    # run simulation
    v, t = run_simulation(problem.cell, **problem.simulation_params)

    print 'Fitness: ' + str(individuals_file.fitness.iloc[best].values)
    print 'Candidate: ' + str(candidate)

    pl.figure()
    pl.plot(data.t, data.v, 'k', label='data to fit')
    pl.plot(t, v, 'r', label='fitted model')
    pl.legend()
    pl.savefig(save_dir+method+'/best_candidate'+str(trial)+'.png')
    pl.show()

    # plot currents
    pl.figure()
    for i, current in enumerate(currents):
        pl.plot(t, -1 * np.array(current), label=channel_list[i])
    pl.legend()
    pl.savefig(save_dir+method+'/best_candidate_currents'+str(trial)+'.png')
    pl.show()

    # plot error development
    if method == 'SA':
        best = individuals_file.fitness.values
        n_generations += 1
    else:
        best = list()
        for i in range(n_generations):
            best.append(individuals_file.fitness.iloc[individuals_file.index[np.logical_and(individuals_file.generation.isin([i]),
                                                 individuals_file.number.isin([0]))]].values[0])
    pl.figure()
    pl.plot(range(n_generations), best)
    pl.xlabel('Number of generations')
    pl.ylabel('Best Error')
    pl.savefig(save_dir+method+'/error'+str(trial)+'.png')
    pl.show()