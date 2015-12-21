# Load a NEURON cell model into python
# 23/7/2012
# (c) 2012, C.Schmidt-Hieber, University College London

import sys
from neuron import h

h.nrn_load_dll("../nrn/mod/i686/.libs/libnrnmech.so.0")
h.xopen("../nrn/cells/stellate-garden.hoc")

def loadcell(sim_settings):
    # load cell
    cell = h.stellate_garden()

    sim_settings.apply(cell)

    return cell



class Settings:
    def __init__(self, parsdict):
        self.parsdict = parsdict
        self.uncTTXDend = True
        self.use_na8st = True
        self.g_nap = 0
        self.uncTTX = False
        self.unc = True

    def apply(self, cell):
        h.distance(sec=cell.soma)
        for ktype in ["kdr", "kap", "km"]:
            h("vShift_%s = %f" % (ktype, self.parsdict["g_k_vshift"]))
        if self.use_na8st:
            h("vShift_na8st = %f" % self.parsdict["g_na_vshift"])

        if self.uncTTXDend:
            self.f_na_soma = 4.5
        else:
            self.f_na_soma = 1.0

        for section in cell.sl:
            section.Ra = self.parsdict["Ra"]

            dist = h.distance(0.5, sec=section)
            if self.uncTTXDend:
                na_point = 0
            else:
                na_point = self.g_na_func(dist)
            for seg in section:
                seg.spines.scale = self.parsdict["scale_spines"]
                seg.pas.e = self.parsdict["e_pas"]
                seg.pas.g = (1.0/self.parsdict["Rm"]) * seg.spines.scale
                if self.use_na8st:
                    seg.na8st.gbar = na_point * seg.spines.scale * 1e3
                else:
                    seg.nax.gbar = na_point * seg.spines.scale
                    seg.nax.vshift = self.parsdict["g_na_vshift"]
                seg.ih.gfastbar = self.parsdict["g_hcnfast"] * seg.spines.scale
                seg.ih.gslowbar = self.parsdict["g_hcnslow"] * seg.spines.scale
                # seg.kslow.gbar = self.g_kslow * seg.spines.scale
                # seg.kfast.gbar = self.g_kfast * seg.spines.scale
                seg.kdr.gkdrbar = self.parsdict["g_kdr"] * seg.spines.scale
                seg.kap.gkabar = self.parsdict["g_kap"] * seg.spines.scale
                seg.km.gbar = self.parsdict["g_km"] * seg.spines.scale
                seg.nap.gbar = self.g_nap * seg.spines.scale

        for section in cell.axo:
            for seg in section:
                seg.spines.scale = 1.0
                if dist < 30:
                    na_axon = self.parsdict["f_axon"] * self.parsdict["g_na"]
                else:
                    na_axon = self.parsdict["g_na"] * 0.1 # to prevent autonomous spikes in the axon
                if self.use_na8st:
                    seg.na8st.gbar = na_axon * seg.spines.scale * 1e3
                else:
                    seg.nax.gbar = na_axon * seg.spines.scale
                seg.ih.gfastbar = self.parsdict["g_hcnfast"] * seg.spines.scale
                seg.ih.gslowbar = self.parsdict["g_hcnslow"] * seg.spines.scale
                # seg.kslow.gbar = self.g_kslow_axon * seg.spines.scale
                # seg.kfast.gbar = self.g_kfast_axon * seg.spines.scale
                seg.kdr.gkdrbar = self.parsdict["f_axon"] * self.parsdict["g_kdr"] * seg.spines.scale
                seg.kap.gkabar = self.parsdict["f_axon"] * self.parsdict["g_kap"] * seg.spines.scale
                seg.km.gbar = self.parsdict["f_axon"] * self.parsdict["g_km"] * seg.spines.scale
                seg.nap.gbar = self.parsdict["f_axon"] * self.g_nap * seg.spines.scale

        for section in cell.som:
            for seg in section:
                seg.spines.scale = 1.0
                if self.use_na8st:
                    seg.na8st.gbar = self.f_na_soma * self.parsdict["g_na"] * seg.spines.scale * 1e3
                else:
                    seg.nax.gbar = self.f_na_soma * self.parsdict["g_na"] * seg.spines.scale
                seg.ih.gfastbar = self.parsdict["g_hcnfast_soma"] * seg.spines.scale
                seg.ih.gslowbar = self.parsdict["g_hcnslow_soma"] * seg.spines.scale
                # seg.kslow.gbar = self.g_kslow_soma * seg.spines.scale
                # seg.kfast.gbar = self.g_kfast_soma * seg.spines.scale
                seg.kdr.gkdrbar = self.parsdict["g_kdr_soma"] * seg.spines.scale
                seg.kap.gkabar = self.parsdict["g_kap_soma"] * seg.spines.scale
                seg.km.gbar = self.parsdict["g_km_soma"] * seg.spines.scale

        if self.uncTTX:
            if self.use_na8st:
                h("forall {gbar_na8st = 0.0}")
            else:
                h("forall {gbar_nax = 0.0}")

        if self.unc:
            cell.Vrest -= 0.45