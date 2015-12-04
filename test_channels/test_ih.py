from __future__ import division
from neuron import h
from model.cell_builder import *
from test_ionchannel import *
from optimization.optimize_passive_point import *

# load NEURON libraries
h.load_file("stdrun.hoc")

# unvariable time step in NEURON
h("""cvode.active(0)""")

__author__ = 'caro'


if __name__ == "__main__":

    # channels
    channel = 'ih'
    gbars = ['gslowbar', 'gfastbar']

    # load mechanisms
    h.nrn_load_dll('./hcn/i686/.libs/libnrnmech.so')

    # parameters
    celsius = 24
    amps = [-60, -110, -60]
    durs = [1000, 1600, 1000]
    v_steps = [-110, -90, -70, -50, -40]
    stepamp = 2
    pos = 0.5
    dt = 0.025

    # try to replicate results from (Fransen, 2004)

    # create Cell
    axon_secs = np.zeros(1, dtype=object)
    dendrites = np.zeros(2, dtype=object)
    soma = Section(geom={'L': 20, 'diam': 15}, Ra=100, cm=1, mechanisms={channel: {},
                        'leak': {}, 'nap': {}, 'pas': {}})
    dendrites[0] = Section(geom={'L': 50, 'diam': 2.2}, Ra=100, cm=1, mechanisms={'leak': {}, 'pas': {}},
                        parent=soma, connection_point=1.0)
    dendrites[1] = Section(geom={'L': 300, 'diam': 1.9}, Ra=100, cm=1, mechanisms={channel: {},
                        'leak': {}, 'nap': {}, 'pas': {}}, parent=soma, connection_point=0.0, nseg=3)
    axon_secs[0] = Section(geom={'L': 400, 'diam': 5.5}, Ra=100, cm=1, mechanisms={channel: {},
                        'leak': {}, 'nap': {}, 'pas': {}}, parent=soma, connection_point=1.0, nseg=2)

    # set equilibrium potential
    for section in [soma, dendrites[0], dendrites[1], axon_secs[0]]:
        if section is not dendrites[0]: setattr(section, 'ena', 87)
        setattr(section, 'e_pas', -83)
        setattr(section, 'g_pas', 1/50000)

    # voltage steps: control
    h.celsius = celsius
    i_steps_control, t = voltage_steps(soma, amps, durs, v_steps, stepamp, pos, dt)

    # voltage steps: blockade
    for section in [soma, dendrites[1], axon_secs[0]]:
        for seg in section:
            setattr(seg.ih, 'gslowbar', 0.0)
            setattr(seg.ih, 'gfastbar', 0.0)
    i_steps_blockade, t = voltage_steps(soma, amps, durs, v_steps, stepamp, pos, dt)

    # plot voltage steps
    pl.figure()
    for step in range(len(v_steps)):
        pl.plot(t, i_steps_control[step] - i_steps_blockade[step], label=str(v_steps[step]), linewidth=2.0)
    pl.legend(loc='lower right')
    pl.xlabel('t (ms)')
    pl.ylabel('i (nA)')
    pl.ylim([-1.2, 0.0])
    pl.xlim([800, t[-1]])
    pl.show()

    # compare to point neuron with no other current
    durs = [100, 1600, 100]
    cell_point = Section(geom={'L': 10, 'diam': 10}, nseg=1, Ra=100, cm=1, mechanisms={channel: {}})
    sec_channel = cell_point(.5).ih
    setattr(cell_point(.5).ih, 'gslowbar', 0.001)
    setattr(cell_point(.5).ih, 'gfastbar', 0.001)

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell_point, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)

    # compute response to voltage steps at different temperature
    celsius = 36
    setattr(cell_point(.5).ih, 'gslowbar', 0.001)
    setattr(cell_point(.5).ih, 'gfastbar', 0.001)

    i_steps, t = current_subtraction(cell_point, sec_channel, gbars, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)