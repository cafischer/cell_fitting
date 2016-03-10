from fit_currents.currentfitting import *
from fit_currents.vclamp import *

__author__ = 'caro'

if __name__ == "__main__":

    # parameters
    cellid = '2015_08_11d'
    objective = 'stepcurrent_test'
    save_dir = './fit_'+cellid+'/'+objective+'/'
    data_dir = '../data/new_cells/'+cellid+'/stepcurrent/stepcurrent-0.1.csv'
    current_dir = './current_traces/'+cellid+'/'+objective+'/'
    model_dir = '../model/cells/point.json'
    mechanism_dir = '../model/channels'
    mechanism_dir_clamp = '../model/channels_without_output'

    channel_list = ['passive', 'ih_slow', 'ih_fast']  #
    E_ion = {'ehcn': -20, 'e_pas': -73.6}  #

    keys = ["ion", "pas", "e_pas"]  # [""]  #
    var_name = keys[-1]
    var_range = np.arange(-71, -69, 0.01)  # [0]  #

    alpha = 0.1
    n_chunks = 10
    fit_cm = True

    # create save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load mechanisms
    if sys.maxsize > 2**32: mechanism_dir += '/x86_64/.libs/libnrnmech.so'
    else: mechanism_dir += '/i686/.libs/libnrnmech.so'
    h.nrn_load_dll(mechanism_dir)
    if sys.maxsize > 2**32: mechanism_dir_clamp += '/x86_64/.libs/libnrnmech.so'
    else: mechanism_dir_clamp += '/i686/.libs/libnrnmech.so'
    h.nrn_load_dll(mechanism_dir_clamp)

    # load data
    data = pd.read_csv(data_dir)
    v = np.array(data.v)
    i_inj = np.array(data.i)
    t = np.array(data.t)

    # estimate derivative via ridge regression
    dt = t[1] - t[0]
    chunk = len(t) / n_chunks
    ds = np.DataSource()
    if ds.exists(save_dir+'/dv.npy'):
        with open(save_dir+'/dv.npy', 'r') as f:
            dv = np.load(f)
    #else:
    #dv = np.zeros(len(t))
    #for i in range(int(len(t)/chunk)):
    #    dv[i*chunk:(i+1)*chunk] = derivative_ridgeregression(v[i*chunk:(i+1)*chunk], dt, alpha)
    dv = np.concatenate((np.array([0]), np.diff(v) / np.diff(t)))

    #pl.figure()
    #pl.plot(t, dv, 'k', linewidth=1.5, label='dV/dt')  # alpha: '+str(alpha) + '\n' + 'n_chunks: ' + str(n_chunks))
    #pl.plot(t, (v-v[0])*np.max(dv)/np.max(v-v[0]), 'b', linewidth=1.5, label='V')
    #pl.xlabel('Time (ms)', fontsize=18)
    #pl.legend(loc='upper right', fontsize=18)
    #pl.show()

    # change parameters and find best fit
    best_fit = [[]] * len(var_range)
    errors = [[]] * len(var_range)

    for i, val in enumerate(var_range):
        print var_name + ': ' + str(val)

        # create cell
        cell = Cell(model_dir)
        cell_area = cell.soma(.5).area() * 1e-8

        # update cell
        cell.update_attr(keys, val)
        E_ion[var_name] = val  # TODO

        # compute currents
        channel_currents = fit_currents.vclamp.vclamp(v, t, cell, channel_list, E_ion)

        # linear regression
        best_fit[i], residual = current_fitting(dvdt, t, copy.deepcopy(i_inj), channel_currents, cell_area,
                                                channel_list, cm=cell.soma.cm, fit_cm=fit_cm, save_dir=save_dir,
                                                plot=False)

        # run simulation to check best fit
        errors[i], _, _ = simulate(best_fit[i], cell, E_ion, data, save_dir, plot=False)

        """
        # plot current trace and derivative
        for j, current in enumerate(channel_currents):
            pl.figure()
            pl.plot(t, dv*cell.soma.cm*cell_area-i_inj*1e-3, 'k', linewidth=1.5, label='$cm \cdot dV/dt - i_{inj}$')
            pl.plot(t, current*np.max(np.abs(dv*cell.soma.cm*cell_area-i_inj*1e-3))/np.max(np.abs(current)),
                    linewidth=1.5, label=channel_list[j])
            pl.ylabel('Current (pA)', fontsize=18)
            pl.xlabel('Time (ms)', fontsize=18)
            pl.ylim([-0.00011, 0.00001])
            pl.legend(loc='upper right', fontsize=18)
            pl.savefig(save_dir+channel_list[j]+'.png')
            pl.show()
        """

    # find best fit
    best = np.argmin(errors)

    print
    print "Best fit over parameter: "
    print var_range[best]
    print best_fit[best]
    print "Error: " + str(errors[best])

    # plot best fit
    cell = Cell(model_dir)
    cell.update_attr(keys, var_range[best])

    # set equilibrium potentials
    for i, (eion, E) in enumerate(E_ion.iteritems()):
        if eion == 'ehcn':
            cell.update_attr(["soma", "mechanisms", "ih", "ehcn"], E)
        elif eion == 'e_pas':
            cell.update_attr(["ion", "pas", eion], E)
        elif 'na' in eion:
            cell.update_attr(["ion", "na_ion", eion], E)
        elif 'k' in eion:
            cell.update_attr(["ion", "k_ion", eion], E)
        elif 'ca' in eion:
            cell.update_attr(["ion", "ca_ion", eion], E)

    simulate(best_fit[best], cell, E_ion, data, save_dir, plot=True)

    # save results
    with open(save_dir+'/best_fit.json', 'w') as f:
        json_utils.dump(best_fit[best], f)
    np.savetxt(save_dir+'/error.json', np.array([errors[best]]))
    cell.save_as_json(save_dir+'cell.json')
    with open(save_dir+'/dv.npy', 'w') as f:
        np.save(f, dvdt)
    np.savetxt(save_dir+'/alpha.txt', np.array([alpha]))
    np.savetxt(save_dir+'/n_chunks.txt', np.array([n_chunks]))