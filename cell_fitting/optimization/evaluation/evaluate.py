import numpy as np
import pandas as pd
import json
from cell_fitting.optimization.fitter import FitterFactory, HodgkinHuxleyFitter
import matplotlib.pyplot as pl
import os
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.read_heka import get_sweep_index_for_amp
pl.style.use('paper')

__author__ = 'caro'


def get_candidate(save_dir, id, generation):
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    idx_bool = np.logical_and(candidates.generation == generation, candidates.id == id)
    candidate = candidates[idx_bool]
    return candidate


def get_best_candidate(save_dir, n_best):
    try:
        if os.stat(save_dir + '/candidates.csv').st_size == 0:  # checks if file is completely empty e.g. when error during optimization occured
            return None
    except OSError:  # if file does not exist
        return None
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    if candidates.empty:
        return None
    candidates_best = pd.DataFrame(columns=candidates.columns)
    for id in np.unique(candidates.id):
        candidates_id = candidates[candidates.id == id]
        candidates_best = candidates_best.append(candidates_id.iloc[np.argmin(candidates_id.fitness.values)])

    idx_best = np.argsort(candidates_best.fitness.values)[n_best]
    best_candidate = candidates_best.iloc[idx_best]
    return best_candidate


def get_best_candidate_new_fitfuns(save_dir, fitter_params):
    try:
        if os.stat(save_dir + '/candidates.csv').st_size == 0:  # checks if file is completely empty e.g. when error during optimization occured
            return None
    except OSError:  # if file does not exist
        return None
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    if candidates.empty:
        return None

    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)

    fitter_params['model_dir'] = optimization_settings['fitter_params']['model_dir']
    fitter_params['mechanism_dir'] = None
    fitter_params['variable_keys'] =  optimization_settings['fitter_params']['variable_keys']
    fitter = HodgkinHuxleyFitter(**fitter_params)
    # candidates = candidates.iloc[:10]
    fitnesses = np.zeros(len(candidates))
    for i in range(len(candidates)):
        fitnesses[i] = fitter.evaluate_fitness(get_candidate_params(candidates.iloc[i]), None)

    idxs = np.argsort(fitnesses)
    return candidates.iloc[idxs], fitnesses[idxs]


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

    for i, sim_params in enumerate(fitter.simulation_params):
        for j in range(len(fitter.fitfuns[i])):
            v_model, t, i_inj = fitter.simulate_cell(best_candidate_params, sim_params)

            pl.figure()
            #if np.size(fitter.data_sets_to_fit[i][j]) == len(t): TODO
                #pl.plot(t, fitter.data_sets_to_fit[i][j], 'k', label='Exp. Data')
            if np.size(fitter.data_sets_to_fit[j]) == len(t):
                pl.plot(t, fitter.data_sets_to_fit[j], 'k', label='Exp. Data')
            pl.plot(t, v_model, 'r', label='Model')
            pl.legend()
            pl.xlabel('Time (ms)')
            pl.ylabel('Membrane Potential (mV)')
            pl.tight_layout()
            pl.savefig(save_dir + 'best_candidate'+str(i)+'.png')
            pl.show()


def plot_best_candidate(save_dir, n_best):
    best_candidate = get_best_candidate(save_dir, n_best)
    plot_candidate(save_dir, best_candidate)
    return best_candidate


def plot_candidate_on_other_data(save_dir, candidate, data_read_dict, plot_dir):

    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)

    optimization_settings['fitter_params']['data_read_dict_per_data_set'] = data_read_dict  # TODO: [data_read_dict]
    optimization_settings['fitter_params']['mechanism_dir'] = None
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    # use saved cell
    fitter.cell = Cell.from_modeldir(os.path.join(save_dir, 'cell.json'))
    fitter.cell.insert_mechanisms(fitter.variable_keys)
    candidate_params = get_candidate_params(candidate)

    v_model, t, i_inj = fitter.simulate_cell(candidate_params, fitter.simulation_params[0])

    if not os.path.exists(os.path.dirname(os.path.join(save_dir, plot_dir))):
        os.makedirs(os.path.dirname(os.path.join(save_dir, plot_dir)))

    pl.figure()
    # if np.size(fitter.data_sets_to_fit[0][0]) == len(t): TODO
    # pl.plot(t, fitter.data_sets_to_fit[0][0], 'k', label='Exp. Data')
    pl.plot(t, v_model, 'r', label='Model')
    pl.legend(fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Membrane Potential (mV)', fontsize=16)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, plot_dir))
    pl.show()


def save_cell(save_dir, candidate):
    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)
    optimization_settings['fitter_params']['mechanism_dir'] = None
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    fitter.cell = Cell.from_modeldir(os.path.join(save_dir, 'cell.json'))  # use saved cell
    fitter.cell.insert_mechanisms(fitter.variable_keys)
    candidate_params = get_candidate_params(candidate)

    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(os.path.join(save_dir))
    with open(os.path.join(save_dir, 'cell.json'), 'w') as f:
        fitter.update_cell(candidate_params)
        cell = fitter.cell.get_dict()
        json.dump(cell, f, indent=4)


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
    save_dir = '../../results/server_17_12_04/2017-12-16_10:04:51/58'
    #save_dir = '../scripts/test/0/'
    # [58, 49, 181, 42, 215, 0, 20, 6, 244, 210]
    #2017-12-16_10:04:51 : 42, 20

    method = 'L-BFGS-B'
    save_dir = os.path.join(save_dir, method)

    best_candidate = plot_best_candidate(save_dir, 0)
    #best_candidate = get_best_candidate(save_dir, 0)
    #load_mechanism_dir('../../model/channels/vavoulis')

    save_cell(save_dir, best_candidate)

    data_read_dict0 = {'data_dir': '/home/cf/Phd/DAP-Project/cell_data/raw_data', 'cell_id': '2013_12_11a',
                      'protocol': 'rampIV', 'sweep_idx': get_sweep_index_for_amp(0.5, 'rampIV'),
                      'v_rest_shift': -8, 'file_type': 'dat'}
    protocol = 'hyperRampTester(3)'
    data_read_dict1 = {'data_dir': '../../data/dat_files', 'cell_id': '2013_12_11a',
                       'protocol': protocol, 'sweep_idx': 0, 'v_rest_shift': -8, 'file_type': 'dat'}
    protocol = 'depoRampTester(3)'
    data_read_dict2 = {'data_dir': '../../data/dat_files', 'cell_id': '2013_12_11a',
                       'protocol': protocol, 'sweep_idx': 0, 'v_rest_shift': -8, 'file_type': 'dat'}
    plot_candidate_on_other_data(save_dir, best_candidate, [data_read_dict1, data_read_dict1, data_read_dict2], 'img/rampIV/0.5(nA).png')

    # data_read_dict = {'data_dir': '/home/cf/Phd/DAP-Project/cell_data/raw_data', 'cell_id': '2013_12_11a',
    #                   'protocol': 'rampIV', 'sweep_idx': get_sweep_index_for_amp(3.1, 'rampIV'),
    #                   'v_rest_shift': -8, 'file_type': 'dat'}
    # plot_candidate_on_other_data(save_dir, best_candidate, data_read_dict, 'img/rampIV/3.1(nA).png')

    data_read_dict = {'data_dir': '/home/cf/Phd/DAP-Project/cell_data/raw_data', 'cell_id': '2013_12_11a',
                      'protocol': 'rampIV', 'sweep_idx': get_sweep_index_for_amp(0.5, 'rampIV'),
                      'v_rest_shift': -8, 'file_type': 'dat'}
    plot_candidate_on_other_data(save_dir, best_candidate, data_read_dict, 'img/rampIV/0.5(nA).png')

    data_read_dict = {'data_dir': '/home/cf/Phd/DAP-Project/cell_data/raw_data', 'cell_id': '2013_12_11a',
                      'protocol': 'plot_IV', 'sweep_idx': get_sweep_index_for_amp(-0.15, 'plot_IV'),
                      'v_rest_shift': -8, 'file_type': 'dat'}
    plot_candidate_on_other_data(save_dir, best_candidate, data_read_dict, 'img/plot_IV/-0.15(nA).png')

    data_read_dict = {'data_dir': '/home/cf/Phd/DAP-Project/cell_data/raw_data', 'cell_id': '2013_12_11a',
                      'protocol': 'plot_IV', 'sweep_idx': get_sweep_index_for_amp(0.1, 'plot_IV'),
                      'v_rest_shift': -8, 'file_type': 'dat'}
    plot_candidate_on_other_data(save_dir, best_candidate, data_read_dict, 'img/plot_IV/0.4(nA).png')

    data_read_dict = {'data_dir': '/home/cf/Phd/DAP-Project/cell_data/raw_data', 'cell_id': '2013_12_11a',
                      'protocol': 'plot_IV', 'sweep_idx': get_sweep_index_for_amp(0.8, 'plot_IV'),
                      'v_rest_shift': -8, 'file_type': 'dat'}
    plot_candidate_on_other_data(save_dir, best_candidate, data_read_dict, 'img/plot_IV/0.8(nA).png')