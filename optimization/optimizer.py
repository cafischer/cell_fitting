from __future__ import division
import pylab as pl
import pandas as pd
import os
from json_utils import *
from neuron import h
from model.cell_builder import *

# load NEURON libraries
h.load_file("stdrun.hoc")

# unvariable time step in NEURON
h("""cvode.active(0)""")

__author__ = 'caro'


def quadratic_error(a, b):
    """
    Computes the quadratic error of the two input arrays.
    :param a: Array containing float numbers.
    :type a: array_like
    :param b: Array containing float numbers.
    :type b: array_like
    :return: Quadratic error of the two input arrays.
    :rtype: float
    """
    return np.sum((a - b)**2)


class Optimizer:
    """
    Can be used to fit parameters of a Cell to experimental data.

    :ivar cell: Cell that shall be fitted to the experimental data.
    :type cell: Cell
    :ivar data: Keys define the objective, values contain DataFrames that are used for the optimization.
    :type data: dict
    :ivar least_error: Keys define the objective, values are used for storing the least error of each generation.
    :type least_error: dict

    Example:
    optimizer = Optimizer(save_dir='./demo/demo_results',
        data_dir={'spike': './demo/demo_data_spike.csv', 'stepcurrent': './demo/demo_data_stepcurrent.csv'},
        model_dir='./demo/demo_cell2.json', mechanism_dir=None,
        objectives=["spike", "stepcurrent"],
        variables=[["hh", 0.01, 0.1, ["soma", "mechanisms", "hh", "gnabar"]], ["Ra", 100, 200, ["soma", "Ra"]]],
        n_gen=10,
        emoo_params={'N': 10, 'C': 200, 'eta_m_0': 20, 'eta_c_0': 20, 'p_m': 0.5})
    """

    def __init__(self, save_dir, data_dir, model_dir, mechanism_dir, objectives, variables, n_gen, emoo_params,
                 simulation_params=None):
        """
        Initializes a Optimizer.

        :param save_dir: Directory for saving.
        :type save_dir: str
        :param data_dir: Keys define the objective, values the corresponding directories to data files.
        :type data_dir: dict
        :param model_dir: Directory to model cell.
        :type model_dir: str
        :param mechanism_dir: Directory to mechanisms of the cell. (see :func Cell.load_mech)
        :type mechanism_dir: str, None
        :param objectives: List of the objectives to be optimized.
        :type objectives: list of str
        :param variables: Contains a list stating the name, lower bound, upper bound, and list of paths (see: :param
        keys in :func Cell.update_attr) for each variable to be optimized.
        :type variables: list of lists each containing str, float, float, list of list of str
        :param emoo_params: Parameters for the evolutionary multi-objective optimization (see parameters of
        :func :Optimizer.init_emoo)
        :type emoo_params: dict
        :param simulation_params: Parameters required for the NEURON simulation (see parameters of
        :func Optimizer.run_simulation)
        :type simulation_params: dict
        """

        # if necessary create saving directory
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # objectives
        self.objectives = objectives

        # variables
        self.variables = variables

        # number of generations
        self.n_gen = n_gen

        # init emoo
        self.emoo = self.init_emoo(**emoo_params)

        # create cell
        self.cell = Cell(model_dir, mechanism_dir)

        # load experimental data
        self.data = dict()
        for obj in objectives:
            self.data[obj] = pd.read_csv(data_dir[obj])

        # simulation parameters
        if simulation_params is None:
            self.simulation_params = self.extract_simulation_params()
        else:
            self.simulation_params = simulation_params

        # initiate error
        self.least_error = dict()
        for obj in objectives:
            self.least_error[obj] = np.zeros(self.n_gen)  # keeps track of error development over generations

        # save parameters
        save_as_json(self.save_dir + '/' + 'objectives.json', self.objectives)
        save_as_json(self.save_dir + '/' + 'variables.json', self.variables)
        save_as_json(self.save_dir + '/' + 'emoo_params.json', emoo_params)
        save_as_json(self.save_dir + '/' + 'simulation_params.json', self.simulation_params, True)
        self.cell.save_as_json(self.save_dir + '/' + 'cell.json')

        # function used after assigning the new variables to recalculate other variables
        self.recalculate_variables = None

    def extract_simulation_params(self):
        """
        Uses the experimental data to extract the simulation parameters.

        :return: Parameters required for the NEURON simulation (see parameters of
        :func Optimizer.run_simulation)
        :rtype: dict
        """
        # load experimental data and simulation parameters
        tstop = dict()
        dt = dict()
        v_init = dict()
        i_amp = dict()
        pos_i = dict()
        pos_v = dict()
        sec = dict()
        for obj in self.objectives:
            self.data[obj].v *= 1000  # convert to mV
            self.data[obj].t *= 1000  # convert to ms
            tstop[obj] = np.array(self.data[obj].t)[-1]
            dt[obj] = np.diff(np.array(self.data[obj].t)[0:2])[0]
            v_init[obj] = np.array(self.data[obj].v)[0]
            i_amp[obj] = np.array(self.data[obj].i)
            pos_i[obj] = 0.5  
            pos_v[obj] = 0.5
            sec[obj] = [self.data[obj].sec[0], self.data[obj].sec[1]]  # [0] section name, [1] section index

        return {'i_amp': i_amp, 'v_init': v_init, 'tstop': tstop, 'dt': dt, 'pos_i': pos_i, 'pos_v': pos_v, 'sec': sec}

    def run_simulation(self, sec, i_amp, v_init, tstop, dt, pos_i, pos_v):
        """
        Runs a NEURON simulation of the cell for the given parameters.

        :param i_amp: Amplitude of the injected current for all times t.
        :type i_amp: array_like
        :param v_init: Initial membrane potential of the cell.
        :type v_init: float
        :param tstop: Duration of a whole run.
        :type tstop: float
        :param dt: Time step.
        :type dt: float
        :param pos_i: Position of the IClamp on the Section (number between 0 and 1).
        :type pos_i: float
        :param pos_v: Position of the recording electrode on the Section (number between 0 and 1).
        :type pos_v: float
        :return: Membrane potential of the cell, time and current amplitude at each time step.
        :rtype: tuple of three ndarrays
        """

        # exchange sec with real Section
        if sec[0] == 'soma':
            section = self.cell.soma
        elif sec[0] == 'dendrites':
            section = self.cell.dendrites[sec[1]]
        else:
            raise ValueError('Given section not defined!')

        # time
        t = np.arange(0, tstop + dt, dt)

        # insert an IClamp with the current trace from the experiment
        stim, i_vec, t_vec = section.play_current(pos_i, i_amp, t)
        i_amp_vec = h.Vector()
        i_amp_vec.record(stim._ref_i)  # record the current amplitude (to check)

        # record the membrane potential
        v = section.record_v(pos_v)

        # run simulation
        h.v_init = v_init
        h.tstop = tstop
        h.steps_per_ms = 1 / dt  # change steps_per_ms before dt, otherwise dt not changed properly
        h.dt = dt
        h.init()
        h.run()
        return np.array(v), t, np.array(i_amp_vec)

    def func_to_optimize(self, variables_new):
        """
        Defines the function to be optimized. It updates the cell with the new variables, runs a simulation and computes
        the quadratic error between the experimental and the simulated membrane potential at each time step.

        :param variables_new: Keys define the variable's name, values the value for this cell.
        :type variables_new: dict
        :return: Keys define the objective, values the corresponding quadratic error for this cell.
        :rtype: dict
        """
        # update the cell with new variables
        for var in self.variables:
            for path in var[3]:
                self.cell.update_attr(path, variables_new[var[0]])
        if self.recalculate_variables is not None:
            self.recalculate_variables(variables_new)

        errors = dict()
        for obj in self.objectives:
            # run simulation with new variables
            v, _, _ = self.run_simulation(**{key: self.simulation_params[key][obj]
                                             for key in self.simulation_params.keys()})

            # compute errors
            errors[obj] = quadratic_error(v, np.array(self.data[obj].v))

        return errors

    def checkpopulation(self, population, columns, gen):
        """
        Defines the function called after each generation. Prints the least error of the population for each objective
        and stores it in self.least_error.

        :param population: Matrix, whereby rows stand for different cells and columns for the values of this cell for
        objectives, variables and emoo parameters.
        :type population: array_like
        :param columns: Contains the indexes of this cell for :param population for objectives, variables and
        emoo parameters.
        :type columns: dict
        :param gen: Index of the current generation.
        :type gen: int
        """

        # look up the errors of the least individual for each objective
        print "Generation: " + str(gen) + "\n Lowest errors are: "
        for obj in self.objectives:
            self.least_error[obj][gen] = np.min(population[:, columns[obj]])

            print obj + ": " + str(self.least_error[obj][gen])

    def init_emoo(self, N, C, eta_m_0, eta_c_0, p_m):
        """
        Initializes the evolutionary multi-objective optimization.

        :param N: Size of the population.
        :type N: int
        :param C: Capacity of the population.
        :type C: int
        :param eta_m_0: Initial strength of mutation  (large values mean weak effect).
        :type eta_m_0: float
        :param eta_c_0: Initial strength of crossover  (large values mean weak effect).
        :type eta_c_0: float
        :param p_m: Probability of mutation of a parameter (holds for each parameter independently).
        :type p_m: float
        :return: Configured emoo ready to use for the evolution.
        :rtype: Emoo
        """

        import emoo

        # initialize emoo
        emoo = emoo.Emoo(N=N, C=C, variables=self.variables, objectives=self.objectives)

        # setup parameters for mutation and crossover
        emoo.setup(eta_m_0=eta_m_0, eta_c_0=eta_c_0, p_m=p_m)

        # define the function to minimize
        emoo.get_objectives_error = self.func_to_optimize

        # define function to be called after each generation
        emoo.checkpopulation = self.checkpopulation

        return emoo

    def run_emoo(self, n_gen):
        # start evolution
        self.emoo.evolution(generations=n_gen)

    def eval_emoo(self, n_inds=10):
        """
        Evaluates emoo:
        - finds the best cell
        - looks up its variables
        - runs simulation with the best cell
        - plots the results of the best cell compared to experiments
        - saves the best variables and the development of the least error.

        :param n_inds: Number of individuals that shall be saved.
        :type n_inds: int
        """
        if self.emoo.master_mode:  # use the master process when multiple processors are used
            population = self.emoo.getpopulation_unnormed() # get the unnormed population
            columns = self.emoo.columns # get the columns vector
        
            # best individuals: sum of all errors smallest
            if n_inds < len(population[:, 0]):
                n_inds = len(population[:, 0])
                print "Less individuals saved due to smaller population!"
            best_inds = np.argsort(np.sum([population[:, columns[obj]] for obj in self.objectives], 0))[:n_inds]
            print "Errors of best individual: "
            for obj in self.objectives:
                print obj + ": " + str(population[best_inds[0], columns[obj]])
        
            # save variables of best individuals
            for i, ind in enumerate(best_inds):
                variables_new = []
                for j, var in enumerate(self.variables):
                    variables_new.append([var[0], population[ind, columns[var[0]]], self.variables[j][3]])  # save
                    # name, value and path
                save_as_json(self.save_dir + '/' + 'variables_new_' + str(i) + '.json', variables_new)

            # update the cell with variables from best individual
            variables_new = load_json(self.save_dir + '/' + 'variables_new_0.json')
            for i, p in enumerate(variables_new):
                for path in self.variables[i][3]:
                    self.cell.update_attr(path, variables_new[i][1])
        
            # run simulation of best individual
            for obj in self.objectives:
                simulation_params_tmp = dict()
                for p in self.simulation_params:
                    simulation_params_tmp[p] = self.simulation_params[p][obj]
                v, _, _ = self.run_simulation(**simulation_params_tmp)
        
                # plot the results
                t = np.arange(0, self.simulation_params['tstop'][obj]+self.simulation_params['dt'][obj],
                              self.simulation_params['dt'][obj])
                pl.figure()
                pl.plot(t, v, 'k', label='model')
                pl.plot(t, self.data[obj].v, label='data')
                pl.xlabel('Time (ms)')
                pl.ylabel('Membrane potential (mV)')
                pl.legend()
                pl.show()

        # save error development
        save_as_json(self.save_dir + '/' + 'least_error.json', self.least_error, True)


########################################################################################################################


def test_extract_simulation_params():
    optimizer = Optimizer(save_dir='../demo/demo_results',
        data_dir={'spike': '../demo/demo_data_spike.csv', 'stepcurrent': '../demo/demo_data_stepcurrent.csv'},
        model_dir='../demo/demo_cell2.json', mechanism_dir=None,
        objectives=["spike", "stepcurrent"],
        variables=[["gnabar", 0.01, 0.1, ["soma", "mechanisms", "hh", "gnabar"]], ["Ra", 100, 200, ["soma", "Ra"]]],
        n_gen=10,
        emoo_params={'N': 10, 'C': 200, 'eta_m_0': 20, 'eta_c_0': 20, 'p_m': 0.5})
    simulation_parameters = optimizer.extract_simulation_params()

    for obj in optimizer.objectives:
        print 'Objective: ' + obj
        print 'v_init: ' + str(simulation_parameters['v_init'][obj])
        print 'pos_i: ' + str(simulation_parameters['pos_i'][obj])
        print 'pos_v: ' + str(simulation_parameters['pos_v'][obj])

        t = np.arange(0, simulation_parameters['tstop'][obj] + simulation_parameters['dt'][obj],
                      simulation_parameters['dt'][obj])
        pl.figure()
        pl.plot(t, simulation_parameters['i_amp'][obj], 'k')
        pl.xlabel('Time (ms)')
        pl.ylabel('Current (nA)')
        pl.title(obj)
        pl.show()


def test_run():
    optimizer = Optimizer(save_dir='../demo/demo_results',
        data_dir={'spike': '../demo/demo_data_spike.csv', 'stepcurrent': '../demo/demo_data_stepcurrent.csv'},
        model_dir='../demo/demo_cell2.json', mechanism_dir=None,
        objectives=["spike", "stepcurrent"],
        variables=[["gnabar", 0.01, 0.1, ["soma", "mechanisms", "hh", "gnabar"]], ["Ra", 100, 200, ["soma", "Ra"]]],
        n_gen=10,
        emoo_params={'N': 10, 'C': 200, 'eta_m_0': 20, 'eta_c_0': 20, 'p_m': 0.5})

    for obj in optimizer.objectives:
        # run of the optimizer .json cell (also runs .hoc cell)
        v, t, i_amp = optimizer.run_simulation(**{key: optimizer.simulation_params[key][obj]
                                             for key in optimizer.simulation_params.keys()})

        # plot the results
        f, (ax1, ax2) = pl.subplots(2, 1, sharex=True)
        ax1.plot(t, optimizer.data[obj].v, 'k', label='data')
        ax1.plot(t, v, 'r', label='model')
        ax1.set_ylabel('Membrane potential (mV)')
        ax1.legend()
        ax2.plot(t, i_amp, 'k')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Current (nA)')
        pl.show()


def test_function_to_optimize():
    optimizer = Optimizer(save_dir='../demo/demo_results',
        data_dir={'spike': '../demo/demo_data_spike.csv', 'stepcurrent': '../demo/demo_data_stepcurrent.csv'},
        model_dir='../demo/demo_cell2.json', mechanism_dir=None,
        objectives=["spike", "stepcurrent"],
        variables=[["gnabar", 0.01, 0.1, [["soma", "mechanisms", "hh", "gnabar"]]], ["Ra", 100, 200, [["soma", "Ra"]]]],
        n_gen=10,
        emoo_params={'N': 10, 'C': 200, 'eta_m_0': 20, 'eta_c_0': 20, 'p_m': 0.5})

    # new variables
    variables_new = {"gnabar": 0.111, "Ra": 122}

    # apply func_to_optimize
    error = optimizer.func_to_optimize(variables_new)

    # test change of attribute
    for i, variable in enumerate(variables_new.keys()):
        if optimizer.cell.get_attr(optimizer.variables[i][3][0]) == variables_new[variable]:
            print "Successful update of the attribute!"
        else:
            print "Attribute not correctly updated!"

    # test error calculation
    for obj in optimizer.objectives:
        v, t, i_amp = optimizer.run_simulation(**{key: optimizer.simulation_params[key][obj]
                                             for key in optimizer.simulation_params.keys()})
        q_error = quadratic_error(v, np.array(optimizer.data[obj].v))
        if q_error == error[obj]:
            print "Correct error computed!"
        else:
            print "Error is not correctly computed!"


def test_evolution():
    optimizer = Optimizer(save_dir='../demo/demo_results',
        data_dir={'spike': '../demo/demo_data_spike.csv', 'stepcurrent': '../demo/demo_data_stepcurrent.csv'},
        model_dir='../demo/demo_cell2.json', mechanism_dir=None,
        objectives=["spike", "stepcurrent"],
        variables=[["gnabar", 0.01, 0.1, [["soma", "mechanisms", "hh", "gnabar"]]], ["Ra", 100, 200, [["soma", "Ra"]]]],
        n_gen=2,
        emoo_params={'N': 10, 'C': 200, 'eta_m_0': 20, 'eta_c_0': 20, 'p_m': 0.5})

    optimizer.run_emoo(optimizer.n_gen)
    optimizer.eval_emoo(2)

if __name__ == "__main__":

    #test_extract_simulation_params()

    #test_run()

    test_function_to_optimize()

    test_evolution()
