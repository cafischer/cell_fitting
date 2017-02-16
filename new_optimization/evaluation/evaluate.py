import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import json
from optimization.errfuns import rms
from new_optimization.fitter import *

__author__ = 'caro'


def plot_candidate(save_dir, id, generation):
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    idx_bool = np.logical_and(candidates.generation == generation, candidates.id == id)
    candidate = candidates.candidate[idx_bool].values[0]
    candidate = np.array([float(x) for x in candidate.split()])

    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)

    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    v_model, t, i_inj = fitter.simulate_cell(candidate)

    pl.figure()
    pl.plot(fitter.data.t, fitter.data.v, 'k', label='Data')
    pl.plot(fitter.data.t, v_model, 'r', label='Model')
    pl.legend(fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Membrane Potential (mV)', fontsize=16)
    pl.savefig(save_dir+'candidate_'+str(id)+'_'+str(generation)+'.png')
    pl.show()


def get_best_candidate(save_dir, n_best):
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    candidates_best = pd.DataFrame(columns=candidates.columns)
    for id in np.unique(candidates.id):
        candidates_id = candidates[candidates.id==id]
        candidates_best = candidates_best.append(candidates_id.iloc[np.argmin(candidates_id.fitness.values)])

    idx_best = np.argsort(candidates_best.fitness.values)[n_best]
    best_candidate = candidates_best.candidate.iloc[idx_best]
    best_candidate = np.array([float(x) for x in best_candidate.split()])
    return best_candidate


def plot_best_candidate(save_dir, n_best):
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    candidates_best = pd.DataFrame(columns=candidates.columns)
    for id in np.unique(candidates.id):
        candidates_id = candidates[candidates.id==id]
        candidates_best = candidates_best.append(candidates_id.iloc[np.argmin(candidates_id.fitness.values)])

    idx_best = np.argsort(candidates_best.fitness.values)[n_best]
    best_candidate = candidates_best.candidate.iloc[idx_best]
    best_candidate = np.array([float(x) for x in best_candidate.split()])
    print 'id: ' + str(candidates_best.id.iloc[idx_best])
    print 'generation: ' + str(candidates_best.generation.iloc[idx_best])
    print 'fitness: ' + str(candidates_best.fitness.iloc[idx_best])
    print 'candidate: ' + str(best_candidate)

    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)

    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    v_model, t, i_inj = fitter.simulate_cell(best_candidate)

    pl.figure()
    pl.plot(fitter.data.t, fitter.data.v, 'k', label='Data')
    pl.plot(fitter.data.t, v_model, 'r', label='Model')
    pl.legend(fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Membrane Potential (mV)', fontsize=16)
    pl.savefig(save_dir+'best_candidate.png')
    pl.show()

    return best_candidate


def plot_candidate_on_other_data(save_dir, candidate, data_dir):

    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)

    optimization_settings['fitter_params']['data_dir'] = data_dir
    optimization_settings['fitter_params']['mechanism_dir'] = None
    fitter = FitterFactory().make_fitter(optimization_settings['fitter_params'])
    v_model, t, i_inj = fitter.simulate_cell(best_candidate)

    pl.figure()
    pl.plot(fitter.data.t, fitter.data.v, 'k', label='Data')
    pl.plot(fitter.data.t, v_model, 'r', label='Model')
    pl.legend(fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Membrane Potential (mV)', fontsize=16)
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
    pl.plot(t, fitter.data.v, 'k', label='Data')
    pl.plot(t, v_model, 'r', label='Model')
    pl.legend(fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Membrane Potential (mV)', fontsize=16)
    pl.title(fitter.fitfun_names[fitfun_id])
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
    save_dir = '../../results/new_optimization/2015_08_06d/15_02_17_PP(4)/'
    #save_dir = '../../results/new_optimization/test/'
    method = 'L-BFGS-B'

    best_candidate = plot_best_candidate(save_dir+method+'/', 0)

    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/raw/rampIV/3.5(nA).csv')
    plot_candidate_on_other_data(save_dir+method+'/', best_candidate, '../../data/2015_08_06d/raw/rampIV/1.0(nA).csv')
    plot_candidate_on_other_data(save_dir+method+'/', best_candidate, '../../data/2015_08_06d/raw/IV/-0.1(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/raw/IV/0.4(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_06d/raw/IV/0.8(nA).csv')
    """
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/corrected_vrest2/rampIV/3.0(nA).csv')
    plot_candidate_on_other_data(save_dir+method+'/', best_candidate, '../../data/2015_08_26b/corrected_vrest2/rampIV/0.5(nA).csv')
    plot_candidate_on_other_data(save_dir+method+'/', best_candidate, '../../data/2015_08_26b/corrected_vrest2/IV/-0.1(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/corrected_vrest2/IV/0.4(nA).csv')
    plot_candidate_on_other_data(save_dir + method + '/', best_candidate, '../../data/2015_08_26b/corrected_vrest2/IV/1.0(nA).csv')
    #plot_min_error_vs_generation(save_dir+method+'/')
    """