from neuron import h, gui
import numpy as np
import csv
h.load_file("stdrun.hoc")  # load NEURON libraries
h("""cvode.active(0)""")  # unvariable time step in NEURON

h.cvode_active(1)
h.cvode.atol(1e-6)

# create simple cell model
cell = h.Section(name='soma')
cell.cm = 1
cell.Ra = 100
cell.diam = 100
cell.L = 100
cell.nseg = 1

# inject current
stim = h.IClamp(cell(0.5))
stim.amp = 1
stim.delay = 10
stim.dur = 50

# insert ionic current
cell.insert('nat')
for seg in cell:
    seg.nat.gbar = 10

# recordings
ina = h.Vector()
ina.record(cell(.5).nat._ref_ina)

v = h.Vector()
v.record(cell(.5)._ref_v)

t = h.Vector()
t.record(h._ref_t)

# run simulation
dt = 0.025
h.tstop = 60
h.steps_per_ms = 1/dt
h.dt = dt
h.run()

# change to arrays
v = np.array(v)
t = np.array(t)
ina = np.array(ina)

# save ina
with open('nat_neuronsim.npy', 'w') as f:
    np.save(f, ina)

# save t, v
header = ['t', 'v']
data_new = np.c_[t, v]
with open('./nat_neuronsim.csv', 'w') as csvoutput:
    writer = csv.writer(csvoutput, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(data_new)

h.load_file("temp.ses")
h.run()
for g in h.List("Graph"):
  #store line, erase, redraw as thick red, and keep on graph thereafter
  xvec = h.Vector()
  yvec = h.Vector()  
  g.getline(-1, xvec,yvec)
  g.erase()
  yvec.line(g, xvec, 2, 1) 
  g.exec_menu("Keep Lines")
  g.exec_menu("Keep Lines")

h.cvode_active(0)
h.run()

import pylab as pl
pl.figure()
pl.show()