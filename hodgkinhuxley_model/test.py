import numpy as np
import matplotlib.pyplot as pl
from model.cell_builder import Cell, Mechanism, complete_mechanismdir
from neuron import h

h.load_file("stdrun.hoc")  # load NEURON libraries

__author__ = 'caro'

save_dir = './test_data/'

cell_params = {'soma': {
    'cm': 1,
    'Ra': 100,
    'geom': {
    'L': 16,
    'diam': 8}}}
cell = Cell(cell_params, complete_mechanismdir('./test_channels/'))
Mechanism('test_channel').insert_into(cell.soma)
cell.soma(.5).test_channel.gbar = 1
cell.soma.ena = 80
Mechanism('leak').insert_into(cell.soma)
cell.soma(.5).leak.gbar = 0.1
cell.soma(.5).leak.e = -65

tstop = 100
dt = 0.01
t = np.arange(0, tstop+dt, dt)
i_inj = np.zeros(len(t))
i_inj[0.02/dt:11/dt] = 1
v0 = -65

stim, i_vec, t_vec = cell.soma.play_current(i_inj, t)
v = cell.soma.record_v()
current_channel = list()
current_channel.append(cell.soma.record_current('leak', ''))
current_channel.append(cell.soma.record_current('test_channel', 'na'))

pg0 = h.Vector()
pg0.record(cell.soma(.5).test_channel._ref_m)
pg1 = h.Vector()
pg1.record(cell.soma(.5).test_channel._ref_h)
minf = h.Vector()
minf.record(cell.soma(.5).test_channel._ref_minf)
hinf = h.Vector()
hinf.record(cell.soma(.5).test_channel._ref_hinf)
mtau = h.Vector()
mtau.record(cell.soma(.5).test_channel._ref_mtau)
htau = h.Vector()
htau.record(cell.soma(.5).test_channel._ref_htau)


h.v_init = v0
h.tstop = tstop
h.steps_per_ms = 1/dt
h.dt = dt
h.run()

pl.figure()
pl.plot(t, v)
pl.show()

pl.figure()
pl.plot(t, current_channel[0], 'b')
pl.plot(t, current_channel[1], 'r')
pl.show()

with open(save_dir+'v.npy', 'w') as f:
    np.save(f, np.array(v))
with open(save_dir+'current_channel.npy', 'w') as f:
    np.save(f, np.array(current_channel))

"""
pl.figure()
pl.plot(t, pg0, 'g')
pl.plot(t, pg1, 'y')
pl.show()


with open('./minf.npy', 'w') as f:
    np.save(f, np.array(minf))
with open('./hinf.npy', 'w') as f:
    np.save(f, np.array(hinf))
with open('./mtau.npy', 'w') as f:
    np.save(f, np.array(mtau))
with open('./htau.npy', 'w') as f:
    np.save(f, np.array(htau))
with open('./pg0.npy', 'w') as f:
    np.save(f, np.array(pg0))
with open('./pg1.npy', 'w') as f:
    np.save(f, np.array(pg1))
"""
