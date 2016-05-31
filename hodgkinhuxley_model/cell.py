from __future__ import division
import numpy as np
import copy

__author__ = 'caro'


class Cell:
    def __init__(self, cm, length, diam, ionchannels):
        self.cm = cm  # (uF/cm2)
        self.length = length  # (um)
        self.diam = diam  # (um)
        self.ionchannels = ionchannels  # list of IonChannels

    def i_inj(self, ts):  # (nA)
        if 10 <= ts <= 20:
            return 1  # TODO
        else:
            return 0

    def derivative_v(self, i_ion, i_inj):
        cell_area = self.length * self.diam * np.pi * 1e-8  # (cm2)
        i_ion = copy.copy(i_ion) * cell_area  # (mA)
        cm = self.cm * cell_area * 1e-3  # (mF)
        i_inj = copy.copy(i_inj) * 1e-6  # (mA)
        return (-1 * np.sum(i_ion, 0) + i_inj) / cm  # (mV/ms)