# Load, manage and print NEURON model parameters
# 23-11-2015
# (c) 2015, C.Schmidt-Hieber, University College London

import sys
import os
import pickle
import numpy as np
from neuron import h
#from schmidthiebermodel import loadcell

TARGET_V = -65.0
DT = 0.02


def i_hold_vclamp(cell, sett, target_v=TARGET_V, use_ttx=False):
    """
    Estimate holding current required to hold the cell at target_v
    by v clamping it to target_v for a long time
    """

    h.tstop = 400.0
    h.steps_per_ms = 1.0/h.dt
    h.dt = DT
    h.v_init = cell.Vrest
    h("forall {nseg=1}")

    if use_ttx:
        if sett.use_na8st:
            h("forall {gbar_na8st = 0.0}")
        else:
            h("forall {gbar_nax = 0.0}")

    vc = h.SEClamp(0.5, sec=cell.soma)
    vc.dur1 = h.tstop
    vc.amp1 = target_v
    vc.rs = 1e-3

    mrec = h.Vector()
    mrec.record(vc._ref_i)
    h.v_init = cell.Vrest
    h.run()

    mrecnp = np.array(mrec)

    return mrecnp[-50.0/DT:].mean()

def read_fi():
    """
    Read best fit parameters
    """
    fsave = open("../dat/min_path", 'r')
    paths_imin = os.path.join("../dat", fsave.read())
    fsave.close()

    fn_minset = paths_imin + "/calibrate_nrn_fi_settings.pck"

    pckf = open(fn_minset, 'rb')
    minset = pickle.load(pckf)
    pckf.close()

    return minset
