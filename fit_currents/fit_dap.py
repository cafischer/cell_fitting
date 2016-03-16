from fit_currents.currentfitting import *
from fit_currents.vclamp import *

__author__ = 'caro'

if __name__ == "__main__":

    # parameters
    cellid = '2015_08_11d'
    objective = 'dap'
    save_dir = './results/fit_'+cellid+'/'+objective+'/'
    data_dir = '../data/new_cells/'+cellid+'/dap/dap.csv'
    model_dir = '../model/cells/point.json'
    passive_weights_dir = './results/fit_'+cellid+'/stepcurrent-0.1/best_fit.json'
    mechanism_dir = '../model/channels_currentfitting'
    mechanism_dir_clamp = '../model/channels_vclamp'
    fit_cm = False

    channel_list = ['na8st', 'nap', 'napsh', 'narsg', 'nat', 'kdr', 'ka', 'km', 'caLVA', 'caHVA', 'kca']
    E_ion = {'ek': -83, 'ena': 87, 'eca': 80}  # eca 90
    E_ion_passive = {'ehcn': -20, 'e_pas': -73.6}  # change for different models!
    C_ion = {'cai': 1e-04, 'cao': 2}   # (Svoboda 2000)

    # parameter to loop over
    keys = [0]  # [["soma", "mechanisms", "narsg", "x6"], ["soma", "mechanisms", "narsg2", "x6"]]
    var_name = keys[-1]
    var_range = [0]  # np.linspace(-1, 10, 10) #np.logspace(5e11, 2e12, 10)

    # load passive model
    with open(passive_weights_dir, 'r') as f:
        weights_passive_dict = json_utils.load(f)
    cm = weights_passive_dict['cm']
    del weights_passive_dict['cm']
    channel_list_passive = weights_passive_dict.keys()
    weights_passive = [weights_passive_dict[channel] for channel in channel_list_passive]

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

    # compute derivative
    dvdt = np.concatenate((np.array([0]), np.diff(v) / np.diff(t)))
    #pl.figure()
    #pl.plot(t, dvdt, 'k', linewidth=1.5, label='dV/dt')  # alpha: '+str(alpha) + '\n' + 'n_chunks: ' + str(n_chunks))
    #pl.plot(t, (v-v[0])*np.max(dvdt)/np.max(v-v[0]), 'b', linewidth=1.5, label='V')
    #pl.xlabel('Time (ms)', fontsize=18)
    #pl.legend(loc='upper right', fontsize=18)
    #pl.show()

    # change parameters and find best fit
    best_fit = [[]] * len(var_range)
    best_fit_all = [[]] * len(var_range)
    errors = [[]] * len(var_range)
    residuals = [[]] * len(var_range)

    for i, val in enumerate(var_range):
        #print var_name + ': ' + str(val)

        # create cell
        cell = Cell(model_dir)
        cell.update_attr(['soma', 'cm'], cm)
        cell_area = cell.soma(.5).area() * 1e-8

        # update cell
        cell.update_attr(keys, val)
        #E_ion[var_name] = val  # TODO

        # compute current
        currents = fit_currents.vclamp.vclamp(v, t, cell, channel_list, E_ion, C_ion)
        currents_passive = fit_currents.vclamp.vclamp(v, t, cell, channel_list_passive, E_ion_passive)

        i_passive = np.dot(weights_passive, np.array(currents_passive) * 1e3 * cell_area)

        # linear regression
        best_fit[i], residuals[i] = current_fitting(dvdt, t, copy.deepcopy(i_inj), copy.deepcopy(currents), cell_area,
                                            channel_list, i_passive=i_passive, cm=cell.soma.cm, fit_cm=fit_cm,
                                            save_dir=save_dir, plot=True)

        # run simulation to check best fit
        E_ion_all = E_ion.copy()
        E_ion_all.update(E_ion_passive)
        best_fit_all[i] = best_fit[i].copy()
        best_fit_all[i].update(weights_passive_dict)
        #errors[i], _, _ = simulate(best_fit_all[i], cell, E_ion_all, data, save_dir, plot=False)

    """
        # plot current trace and derivative
        for j, current in enumerate(currents):
            pl.figure()
            pl.plot(t, dvdt*cell.soma.cm*cell_area-i_inj*1e-3, 'k', linewidth=1.5, label='$cm \cdot dV/dt - i_{inj}$')
            pl.plot(t, -1 * current*np.max(np.abs(dvdt*cell.soma.cm*cell_area-i_inj*1e-3))/np.max(np.abs(current)),
                    linewidth=1.5, label=channel_list[j])
            pl.ylabel('Current (pA)', fontsize=18)
            pl.xlabel('Time (ms)', fontsize=18)
            #pl.ylim([-0.00011, 0.00001])
            pl.legend(loc='upper right', fontsize=18)
            pl.savefig(save_dir+channel_list[j]+'.png')
            pl.show()
"""

    # find best fit
    best = np.argmin(residuals)

    print
    print "Best fit over parameter: "
    print var_range[best]
    print best_fit[best]
    print "Error: " + str(errors[best])

    # integrate linear regression solution
    weights_active = [best_fit[best][channel] for channel in channel_list]
    for i in range(len(currents)):
        currents[i] *= -1 * 1e3 * cell_area
    dvdt_fromfit = (np.dot(weights_active, currents) - i_passive + i_inj * 1e-3) / (cm * cell_area)
    dt = t[1] - t[0]
    v_int = np.zeros(len(v))
    v_int[0] = v[0]
    for i in range(1, len(v)):
        v_int[i] = v_int[i-1] + dt * dvdt_fromfit[i-1]

    #pl.figure()
    #pl.plot(t, v, 'k', label='V from data')
    #pl.plot(t, v_int, label='V from linear regression')
    #pl.ylabel('V (mV)')
    #pl.xlabel('Time (ms)')
    #pl.legend()
    #pl.show()

    # plot best fit
    cell = Cell(model_dir)
    cell.update_attr(['soma', 'cm'], cm)
    cell.update_attr(keys, var_range[best])

    simulate(best_fit_all[best], cell, E_ion_all, data, C_ion, save_dir=save_dir, plot=True)

    # save results
    with open(save_dir+'/best_fit.json', 'w') as f:
        json_utils.dump(best_fit[best], f)
    with open(save_dir+'/best_fit_all.json', 'w') as f:
        json_utils.dump(best_fit_all[best], f)
    np.savetxt(save_dir+'/error.json', np.array([errors[best]]))
    cell.save_as_json(save_dir+'cell.json')
    with open(save_dir+'/dvdt.npy', 'w') as f:
        np.save(f, dvdt)