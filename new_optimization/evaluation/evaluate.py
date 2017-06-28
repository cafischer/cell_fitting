import numpy as np
import pandas as pd
import json
from new_optimization.fitter import FitterFactory
import matplotlib.pyplot as pl
import os
from nrn_wrapper import Cell

__author__ = 'caro'


def get_candidate(save_dir, id, generation):
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    idx_bool = np.logical_and(candidates.generation == generation, candidates.id == id)
    candidate = candidates[idx_bool]
    return candidate


def get_best_candidate(save_dir, n_best):
    if os.stat(save_dir + '/candidates.csv').st_size == 0:  # checks if file is completely empty e.g. when error during optimization occured
        return None
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    if not candidates.empty:
        candidates_best = pd.DataFrame(columns=candidates.columns)
        for id in np.unique(candidates.id):
            candidates_id = candidates[candidates.id == id]
            candidates_best = candidates_best.append(candidates_id.iloc[np.argmin(candidates_id.fitness.values)])

        idx_best = np.argsort(candidates_best.fitness.values)[n_best]
        best_candidate = candidates_best.iloc[idx_best]
        return best_candidate


def get_candidate_params(candidate):
    candidate_params = candidate.candidate
    candidate_params = np.array([float(x) for x in candidate_params.split()])
    return candidate_params


def plot_candidate(save_dir, candidate):

    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])

    best_candidate_params = get_candidate_params(candidate)
    print 'id: ' + str(candidate.id)
    print 'generation: ' + str(candidate.generation)
    print 'fitness: ' + str(candidate.fitness)
    for k, v in zip(fitter.variable_keys, best_candidate_params):
        print k, v

    # use saved cell
    fitter.cell = Cell.from_modeldir(os.path.join(save_dir, 'cell.json'))
    fitter.cell.insert_mechanisms(fitter.variable_keys)

    if type(fitter.simulation_params) is list:
        for i, sim_params in enumerate(fitter.simulation_params):
            v_model, t, i_inj = fitter.simulate_cell(best_candidate_params, sim_params)
            pl.figure()
            pl.plot(fitter.datas[i].t, fitter.datas[i].v, 'k', label='Exp. Data')
            pl.plot(fitter.datas[i].t, v_model, 'r', label='Model')
            pl.legend(fontsize=16)
            pl.xlabel('Time (ms)', fontsize=16)
            pl.ylabel('Membrane Potential (mV)', fontsize=16)
            pl.tight_layout()
            pl.savefig(save_dir + 'best_candidate'+str(i)+'.png')
            pl.show()
    else:
        v_model, t, i_inj = fitter.simulate_cell(best_candidate_params)

        pl.figure()
        pl.plot(fitter.data.t, fitter.data.v, 'k', label='Exp. Data')
        pl.plot(fitter.data.t, v_model, 'r', label='Model')
        pl.legend(fontsize=16)
        pl.xlabel('Time (ms)', fontsize=16)
        pl.ylabel('Membrane Potential (mV)', fontsize=16)
        pl.tight_layout()
        pl.savefig(save_dir+'best_candidate.png')
        pl.show()


def plot_best_candidate(save_dir, n_best):
    best_candidate = get_best_candidate(save_dir, n_best)
    plot_candidate(save_dir, best_candidate)
    return best_candidate


def plot_candidate_on_other_data(save_dir, candidate, data_dir):

    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)

    if (optimization_settings['fitter_params']['name'] == 'HodgkinHuxleyFitterSeveralData' or
        optimization_settings['fitter_params']['name'] == 'HodgkinHuxleyFitterSeveralDataAdaptive'):
        optimization_settings['fitter_params']['data_dirs'] = [data_dir]
        optimization_settings['fitter_params']['mechanism_dir'] = None
        fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
        # use saved cell
        fitter.cell = Cell.from_modeldir(os.path.join(save_dir, 'cell.json'))
        fitter.cell.insert_mechanisms(fitter.variable_keys)
        v_model, t, i_inj = fitter.simulate_cell(get_candidate_params(candidate), fitter.simulation_params[0])
    else:
        optimization_settings['fitter_params']['data_dir'] = data_dir
        optimization_settings['fitter_params']['mechanism_dir'] = None
        fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
        # use saved cell
        fitter.cell = Cell.from_modeldir(os.path.join(save_dir, 'cell.json'))
        fitter.cell.insert_mechanisms(fitter.variable_keys)
        candidate_params = get_candidate_params(candidate)

        #fitter.simulation_params['v_init'] = -70  # TODO
        #vshift = -15.5
        #candidate_params[5:10] += vshift  # TODO!
        #fitter.cell.soma.ena += vshift
        #fitter.cell.soma.ek += vshift
        #fitter.cell.soma.e_pas += vshift
        #print 'ena', fitter.cell.soma.ena
        #print 'epas', fitter.cell.soma.e_pas
        #candidate_params[1] = -85

        #with open('../../model/cells/test.json', 'w') as f:
        #    fitter.update_cell(candidate_params)
        #    cell = fitter.cell.get_dict()
        #    json.dump(cell, f, indent=4)

        v_model, t, i_inj = fitter.simulate_cell(candidate_params)

    pl.figure()
    pl.plot(fitter.data.t, fitter.data.v, 'k', label='Exp. Data')
    pl.plot(fitter.data.t, v_model, 'r', label='Model')
    pl.legend(fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Membrane Potential (mV)', fontsize=16)
    pl.tight_layout()
    pl.show()


def plot_min_error_vs_generation(save_dir):
    candidates = pd.read_csv(save_dir + '/candidates.csv')

    best_fitnesses = list()
    for generation in range(candidates.generation.iloc[-1]+1):
        candidates_generation = candidates[candidates.generation == generation]
        best_fitnesses.append(candidates_generation.fitness[np.argmin(candidates_generation.fitness)])

    pl.figure()
    pl.plot(range(candidates.generation.iloc[-1]+1), best_fitnesses, 'k')
    pl.xlabel('Generation')
    pl.ylabel('Error')
    pl.tight_layout()
    pl.savefig(save_dir+'error_development.png')
    pl.show()


def plot_best_candidate_severalfitfuns(save_dir, fitfun_id):
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    candidates.fitness = candidates.fitness.apply(lambda x: [float(n) for n in x.replace(',', '').replace('[', '').replace(']', '').split()])
    candidates.fitness = candidates.fitness.apply(lambda x: x[fitfun_id])
    best_candidate = candidates.candidate[np.argmin(candidates.fitness)]
    best_candidate = np.array([float(x) for x in best_candidate.split()])

    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)

    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    v_model, t, i_inj = fitter.simulate_cell(best_candidate)

    pl.figure()
    pl.plot(t, fitter.data.v, 'k', label='Exp. Data')
    pl.plot(t, v_model, 'r', label='Model')
    pl.legend(fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Membrane Potential (mV)', fontsize=16)
    pl.title(fitter.fitfun_names[fitfun_id])
    pl.tight_layout()
    pl.savefig(save_dir + 'best_candidate_'+fitter.fitfun_names[fitfun_id]+'.png')
    pl.show()


def plot_min_error_vs_generation_severalfitfuns(save_dir):
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    candidates.fitness = candidates.fitness.apply(
        lambda x: [float(n) for n in x.replace(',', '').replace('[', '').replace(']', '').split()])
    candidates.fitness = candidates.fitness.apply(lambda x: np.mean(x))

    best_fitnesses = list()
    for generation in range(candidates.generation.iloc[-1] + 1):
        candidates_generation = candidates[candidates.generation == generation]
        best_fitnesses.append(candidates_generation.fitness[np.argmin(candidates_generation.fitness)])

    pl.figure()
    pl.plot(range(candidates.generation.iloc[-1] + 1), best_fitnesses, 'k')
    pl.xlabel('Generation')
    pl.ylabel('Error')
    pl.tight_layout()
    pl.savefig(save_dir + 'error_development.png')
    pl.show()


def get_channel_params(channel_name, candidate, save_dir):
    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)

    channel_params = list()
    for i, variable_keys in enumerate(optimization_settings['fitter_params']['variable_keys']):
        for variable_key in variable_keys:
            if channel_name in variable_key:
                channel_params.append((candidate[i], variable_key))
    return channel_params


if __name__ == '__main__':
    #save_dir = '../../results/server/2017-06-23_08:31:00/114/'
    save_dir = '../../results/server/2017-06-19_13:12:49/189/'
    #save_dir = '../../results/optimization_vavoulis_channels/2015_08_26b/22_01_17_readjust1_adaptive/'
    method = 'L-BFGS-B'

    best_candidate = plot_best_candidate(save_dir+method+'/', 0)

    #plot_candidate_on_other_data(save_dir + method + '/', best_candidate,
    #                             '../../data/2015_08_06d/correct_vrest_-16mV/PP(4)/0(nA).csv')
    #plot_candidate_on_other_data(save_dir + method + '/', best_candidate,
    #                             '../../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(3)/0(nA).csv')
    #plot_candidate_on_other_data(save_dir + method + '/', best_candidate,
    #                             '../../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(21)/0(nA).csv')
    """
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/correct_vrest_-16mV/rampIV/3.5(nA).csv')
    plot_candidate_on_other_data(save_dir + method +'/', best_candidate, '../../data/2015_08_06d/correct_vrest_-16mV/rampIV/1.0(nA).csv')
    plot_candidate_on_other_data(save_dir + method +'/', best_candidate, '../../data/2015_08_06d/correct_vrest_-16mV/IV/-0.1(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/correct_vrest_-16mV/IV/0.4(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/correct_vrest_-16mV/IV/0.7(nA).csv')


    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/raw/rampIV/3.0(nA).csv')
    plot_candidate_on_other_data(save_dir+method+'/', best_candidate, '../../data/2015_08_26b/raw/rampIV/0.5(nA).csv')
    plot_candidate_on_other_data(save_dir+method+'/', best_candidate, '../../data/2015_08_26b/raw/IV/-0.1(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/raw/IV/0.4(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/raw/IV/1.0(nA).csv')
    #plot_min_error_vs_generation(save_dir+method+'/')
    """
    """
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/vrest-60/rampIV/3.5(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/vrest-60/rampIV/0.5(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/vrest-60/IV/-0.1(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/vrest-60/IV/0.4(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/vrest-60/IV/1.0(nA).csv')
    """
    """
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/vrest-80/rampIV/3.5(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/vrest-80/rampIV/0.5(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/vrest-80/IV/-0.1(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/vrest-80/IV/0.4(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/vrest-80/IV/1.0(nA).csv')
    """
    """
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/vrest-60/rampIV/3.0(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/vrest-60/rampIV/0.5(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/vrest-60/IV/-0.1(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/vrest-60/IV/0.4(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/vrest-60/IV/1.0(nA).csv')
    """
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/vrest-75/rampIV/0.5(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/vrest-75/IV/-0.1(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/vrest-75/IV/0.4(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/vrest-75/IV/1.0(nA).csv')
