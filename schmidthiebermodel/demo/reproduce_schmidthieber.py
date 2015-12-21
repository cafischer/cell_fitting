import numpy as np
import pylab as pl
import pandas as pd
from json_utils import *
from neuron import h
h.load_file("stdrun.hoc")
from schmidthiebermodel import loadcell

dt = 0.05
h.tstop = 440
h.steps_per_ms = 1/dt
h.dt = dt

# load cell
parsdict = load_json('settings.json')
sim_settings = loadcell.Settings(parsdict)
cell = loadcell.loadcell(sim_settings)

from schmidthiebermodel import calibrate
ihold = calibrate.i_hold_vclamp(cell, sim_settings)

ic_hold = h.IClamp(.5, sec=cell.soma)
ic_hold.delay = 0.0
ic_hold.dur = h.tstop
ic_hold.amp = ihold

clamp = h.IClamp(.5, sec=cell.soma)
clamp.delay = 250
clamp.dur = 190
clamp.amp = 0.45

v = h.Vector()
v.record(cell.soma(.5)._ref_v)

h.run()

t = np.arange(0, h.tstop+h.dt, h.dt)

pl.figure()
pl.plot(t, np.array(v), label='model')
pl.ylabel('Membrane potential (mV)')
pl.xlabel('Time (ms)')
pl.legend()
pl.xlim([300,400])
pl.show()
