from data.hekareader import *
import pandas as pd
import os

if __name__ == '__main__':

    cell = '2015_08_26b'
    file_dir = './'+cell+'/'+cell+'.dat'
    vrest = -62.5

    hekareader = HekaReader(file_dir)
    type_to_index = hekareader.get_type_to_index()

    group = 'Group1'
    protocol = 'rampIV'
    trace = 'Trace1'
    protocol_to_series = hekareader.get_protocol(group)
    series = protocol_to_series[protocol]
    sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series]))]
    sweep_idx = 20
    sweeps = [sweeps[sweep_idx]]

    indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]

    fig = pl.figure()
    ax = fig.add_subplot(111)
    for index in indices:
        x, y = hekareader.get_xy(index)
        x *= 1000
        y *= 1000
        y = y - (y[0] - vrest)
        x_unit, y_unit = hekareader.get_units_xy(index)

        ax.plot(x, y, 'k')
        ax.set_xlabel('Time (ms)', fontsize=18)
        ax.set_ylabel('Membrane Potential (mV)', fontsize=18)
        ax.tick_params(labelsize=15)
    pl.tight_layout()
    pl.show()


    i_inj = pd.read_csv('./Protocols/'+protocol+'.csv', header=None)
    i_inj = np.array(i_inj)[:, 0]
    if protocol == 'IV':
        amp = -0.15 + sweep_idx * 0.05  # for IV
        amp_change = amp
    elif protocol == 'rampIV':
        amp = sweep_idx * 0.1  # for rampIV
        amp_change = amp / 0.1  # for rampIV
    print amp
    i_inj *= amp_change

    data = pd.DataFrame({'v': y, 't': x, 'i': i_inj})

    save_dir = './'+cell+'/'+protocol
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data.to_csv(save_dir+'/'+str(amp)+'(nA).csv', index=False)