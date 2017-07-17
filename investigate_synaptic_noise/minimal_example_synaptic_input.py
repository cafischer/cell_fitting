from neuron import h
import numpy as np
import pylab as pl
h.load_file("stdrun.hoc")  # load NEURON libraries
h("""cvode.active(1)""")  # unvariable time step in NEURON

# create simple cell model
cell = h.Section(name='soma')
cell.cm = 1
cell.Ra = 100
cell.diam = 100
cell.L = 100
cell.nseg = 1
cell.insert('pas')

# insert synapse
syn1 = h.ExpSyn(.5, sec=cell)
syn1.tau = 1  # ms
syn1.e = 0  # mV
syn2 = h.Exp2Syn(.5, sec=cell)
syn2.tau1 = 1  # ms
syn2.tau2 = 2
syn2.e = 0  # mV

h.nrn_load_dll("/home/cf/Phd/programming/projects/bac_project/bac_project/connectivity/vecstim/x86_64/.libs/libnrnmech.so")
spiketimes = [100]  # ms
spiketimes_vec = h.Vector()
spiketimes_vec.from_python(spiketimes)  # convert spiketimes to neuron vector

# make stimulus
stim = h.VecStim(.5, sec=cell)
stim.play(spiketimes_vec)

con1 = h.NetCon(stim, syn1, sec=cell)
con1.weight[0] = 1
con1.delay = 0

con2 = h.NetCon(stim, syn2, sec=cell)
con2.weight[0] = 1
con2.delay = 0


# recordings
i_syn1 = h.Vector()
i_syn1.record(syn1._ref_i)
i_syn2 = h.Vector()
i_syn2.record(syn2._ref_i)
t_rec = h.Vector()
t_rec.record(h._ref_t)

# run simulation
h.tstop = 200
h.dt = 0.1
h.init()
h.run()

# change to arrays
i_syn1 = np.array(i_syn1)
i_syn2 = np.array(i_syn2)
t_rec = np.array(t_rec)

# plot
pl.figure()
pl.plot(t_rec, i_syn1, 'r')
pl.plot(t_rec, i_syn2, 'b')
pl.show()