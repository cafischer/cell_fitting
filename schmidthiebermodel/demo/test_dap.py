import numpy as np
import pylab as pl
import pandas as pd
from json_utils import *
from neuron import h
h.load_file("stdrun.hoc")
from schmidthiebermodel import loadcell


# compare experimental dap with model
data = pd.read_csv('../../data/cell_2013_12_13f/dap/dap_1.csv').convert_objects(convert_numeric=True)
v_exp = data.as_matrix(['v'])
t_exp = data.as_matrix(['t'])
i_exp = data.as_matrix(['i'])

dt = t_exp[1] - t_exp[0]
h.tstop = t_exp[-1]
h.steps_per_ms = 1/dt
h.dt = dt

# load cell
parsdict = load_json('settings.json')
sim_settings = loadcell.Settings(parsdict)
cell = loadcell.loadcell(sim_settings)

stim = h.IClamp(.5, sec=cell.soma)
stim.delay = 0  # 0 necessary for playing the current into IClamp
stim.dur = 1e9  # 1e9 necessary for playing the current into IClamp
i_vec = h.Vector()
i_vec.from_python(i_exp)
t_vec = h.Vector()
t_vec.from_python(np.concatenate((np.array([0]), t_exp)))
i_vec.play(stim._ref_amp, t_vec)

# holding Iclamp
ic_hold = h.IClamp(.5, sec=cell.soma)
ic_hold.delay = 0.0
ic_hold.dur = h.tstop
ic_hold.amp = -0.2

v = h.Vector()
v.record(cell.soma(.5)._ref_v)

v_axon = h.Vector()
v_axon.record(cell.axon(.5)._ref_v)

# measure currents
i_na8st = h.Vector()
i_na8st.record(cell.soma(.5).na8st._ref_ina)
i_ihs = h.Vector()
i_ihs.record(cell.soma(.5).ih._ref_i)
i_ihf = h.Vector()
i_ihf.record(cell.soma(.5).ih._ref_i)
i_km = h.Vector()
i_km.record(cell.soma(.5).km._ref_ik)
i_kdr = h.Vector()
i_kdr.record(cell.soma(.5).kdr._ref_ik)
i_kap = h.Vector()
i_kap.record(cell.soma(.5).kap._ref_ik)

# measure conductances
g_na8st = h.Vector()
g_na8st.record(cell.soma(.5).na8st._ref_g)
g_ihs = h.Vector()
g_ihs.record(cell.soma(.5).ih._ref_gslow)
g_ihf = h.Vector()
g_ihf.record(cell.soma(.5).ih._ref_gfast)
g_km = h.Vector()
g_km.record(cell.soma(.5).km._ref_gk)
g_kdr = h.Vector()
g_kdr.record(cell.soma(.5).kdr._ref_gkdr)
g_kap = h.Vector()
g_kap.record(cell.soma(.5).kap._ref_gka)


vs = []
gs = [0]
for g in gs:
    # add kdr to model cell
    for section in [cell.soma, cell.axon]:
        for seg in section:
            #seg.na8st.gbar = g
            print''
    h.run()

    vs.append(np.array(v))

pl.figure()
pl.plot(t_exp, v_exp, 'k', linewidth=1.5, label='experiment')
for i, v_i in enumerate(vs):
    pl.plot(t_exp, v_i, linewidth=1.5, label='model with g: '+str(gs[i]))
pl.plot(t_exp, np.array(v_axon), 'r', linewidth=1.5, label='model axon')
pl.ylabel('Membrane potential (mV)')
pl.xlabel('Time (ms)')
pl.legend()
pl.show()

#h.run()

t = np.arange(0, h.tstop+h.dt, h.dt)

pl.figure()
pl.plot(t, np.array(v), label='model')
pl.plot(t_exp, v_exp, label='experiment')
pl.ylabel('Membrane potential (mV)')
pl.xlabel('Time (ms)')
pl.legend()
pl.show()


# plot the results
f, (ax1, ax3) = pl.subplots(2, 1, sharex=True)
ax1.plot(t, np.array(v), 'k')
ax1.set_ylabel('Membrane\npotential (mV)')
#ax1.set_xlim([601,603])
ax3.plot(t, np.array(i_na8st), label='i_na8st')
ax3.plot(t, np.array(i_ihs), label='i_ihs')
ax3.plot(t, np.array(i_ihf), label='i_ihf')
ax3.plot(t, np.array(i_km), label='i_km')
ax3.plot(t, np.array(i_kdr), label='i_kdr')
ax3.plot(t, np.array(i_kap), label='i_kap')
ax3.set_ylabel('Current (mA/cm2)')
ax3.set_xlabel('Time (ms)')
ax3.legend()
#ax3.set_xlim([601,603])
pl.show()


# plot the results
f, (ax1, ax2) = pl.subplots(2, 1, sharex=True)
ax1.plot(t, np.array(v), 'k')
ax1.set_ylabel('Membrane\npotential (mV)')
#ax1.set_xlim([601,603])
ax2.plot(t, np.array(g_na8st)/1000, label='g_na8st')
ax2.plot(t, np.array(g_ihs), label='g_ihs')
ax2.plot(t, np.array(g_ihf), label='g_ihf')
ax2.plot(t, np.array(g_km)/10000, label='g_km')
ax2.plot(t, np.array(g_kdr), label='g_kdr')
ax2.plot(t, np.array(g_kap), label='g_kap')
ax2.set_ylabel('Conductance (S/cm2)')
ax2.set_xlabel('Time (ms)')
ax2.legend()
#ax2.set_xlim([601,603])
pl.show()

f, (ax1, ax4) = pl.subplots(2, 1, sharex=True)
ax1.plot(t, np.array(v), label='model')
ax1.plot(t_exp, v_exp, label='experiment')
ax1.set_ylabel('Membrane\npotential (mV)')
ax1.legend()
#ax1.set_xlim([601,603])
ax4.plot(t, i_exp)
ax4.set_ylabel('Injected current (nA)')
ax4.set_xlabel('Time (ms)')
#ax4.set_xlim([601,603])
pl.show()


