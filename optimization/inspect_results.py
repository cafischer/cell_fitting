from __future__ import division
from optimize_passive_full import *

__author__ = 'caro'


if __name__ == "__main__":
    # get optimizer
    optimizer = optimize_passive_full()

    # update the cell with variables from best individual
    for ind in range(100):
        variables_new = load_json(optimizer.save_dir + '/' + 'variables_new_'+str(ind)+'.json')
        for i, p in enumerate(variables_new):
            for path in optimizer.variables[i][3]:
                optimizer.cell.update_attr(path, variables_new[i][1])

        # run simulation of best individual
        for obj in optimizer.objectives:
            simulation_params_tmp = dict()
            for p in optimizer.simulation_params:
                simulation_params_tmp[p] = optimizer.simulation_params[p][obj]

            # run simulation and compute the variable to fit
            var_to_fit, x = optimizer.get_var_to_fit[obj](**simulation_params_tmp)

            # data to fit
            data_to_fit = np.array(optimizer.data[obj][optimizer.var_to_fit[obj]])  # convert to array
            data_to_fit = data_to_fit[~np.isnan(data_to_fit)]  # get rid of nans

            # plot the results
            t = np.arange(0, optimizer.simulation_params['tstop'][obj]+optimizer.simulation_params['dt'][obj],
                          optimizer.simulation_params['dt'][obj])
            pl.figure()
            pl.plot(x, var_to_fit, 'k', label='model')
            pl.plot(x, data_to_fit, label='data')
            pl.ylabel(obj)
            pl.legend()
            pl.show()