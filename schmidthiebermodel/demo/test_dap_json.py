import numpy as np
import pylab as pl
import pandas as pd
from neuron import h
h.load_file("stdrun.hoc")
from model.cell_builder import *

cell = Cell("../../model/cells/morph_json.json", "../nrn/mod/i686/.libs/libnrnmech.so.0")  # stellate_garden


# apply parameter changes
h.celsius = 35
k_vshift = 3.4639
na_vshift = 16.9782

for sec in [cell.soma]+cell.dendrites+cell.axon_secs:
    sec.Ra = 238.6966
    sec.cm = 1.0
    sec.nseg = 5
    Mechanism("pas").insert_into(sec)
    sec.g_pas = 3.6149e-05
    sec.e_pas = -89.0547
    Mechanism("ih").insert_into(sec)
    Mechanism("km").insert_into(sec)
    Mechanism("kdr").insert_into(sec)
    Mechanism("kap").insert_into(sec)
    Mechanism("na8st").insert_into(sec)
    for seg in sec:
        seg.na8st.gbar = 0.0
        seg.kdr.gkdrbar = 0.001478
        seg.kap.gkabar = 0.0679
        seg.km.gbar = 5.5161
        seg.ih.gfastbar = 2.2827e-04
        seg.ih.gslowbar = 1.04609e-04

        h.vShift_kdr = k_vshift
        h.vShift_kap = k_vshift
        h.vShift_km = k_vshift
        h.vShift_na8st = na_vshift
    sec.ena = 60.0
    sec.ek = -85.0

for sec in cell.axon_secs:
    for seg in sec:
        seg.na8st.gbar = 5.2999
        seg.kdr.gkdrbar = 2.4008e-03
        seg.kap.gkabar = 1.1034e-01
        seg.km.gbar = 8.9598
        seg.ih.gfastbar = 2.2518e-04
        seg.ih.gslowbar = 1.0319e-04

for seg in cell.soma:
    seg.na8st.gbar = 238.4942
    seg.kdr.gkdrbar = 9.5404e-05
    seg.kap.gkabar = 0.3606
    seg.km.gbar = 4.7989
    seg.ih.gfastbar = 9.8e-05
    seg.ih.gslowbar = 5.3e-05

# compare experimental dap with model
data = pd.read_csv('../../data/new_cells/2015_08_11d/dap/dap.csv').convert_objects(convert_numeric=True)
v_exp = data.as_matrix(['v'])
t_exp = data.as_matrix(['t'])
i_exp = data.as_matrix(['i'])

dt = t_exp[1] - t_exp[0]
h.tstop = t_exp[-1]
h.steps_per_ms = 1/dt
h.dt = dt

# insert IClamp
stim = h.IClamp(.5, sec=cell.soma)
stim.delay = 0  # 0 necessary for playing the current into IClamp
stim.dur = 1e9  # 1e9 necessary for playing the current into IClamp
i_vec = h.Vector()
i_vec.from_python(i_exp)
t_vec = h.Vector()
t_vec.from_python(np.concatenate((np.array([0]), t_exp)))
i_vec.play(stim._ref_amp, t_vec)

v = h.Vector()
v.record(cell.soma(.5)._ref_v)

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

# run simulation
h.run()

t = np.arange(0, h.tstop+h.dt, h.dt)

# plot V
pl.figure()
pl.plot(t_exp, v_exp, 'k', label='experiment')
pl.plot(t, np.array(v), 'r', label='model')
pl.ylabel('Membrane potential (mV)')
pl.xlabel('Time (ms)')
pl.legend()
pl.show()

# plot currents
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

# plot conductances
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

# plot V, injected current
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
