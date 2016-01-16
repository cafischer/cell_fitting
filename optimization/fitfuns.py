import numpy as np
from neuron import h

h.load_file("stdrun.hoc")  # load NEURON libraries
h("""cvode.active(0)""")  # invariable time step in NEURON

__author__ = 'caro'


def run_impedance(self, cell, sec, i_amp, v_init, tstop, dt, pos_i, pos_v, onset, f_range):
    # f_range = [data['impedance'].f_range[0], data['impedance'].f_range[1]]
    i_amp = np.concatenate((np.zeros(onset/dt), i_amp))
    v, t, i = self.run_simulation(cell, sec, i_amp, v_init, tstop + onset, dt, pos_i, pos_v)
    imp, freqs = self.impedance(v[onset/dt:], i[onset/dt:], dt/1000, f_range)
    return imp, freqs


def run_simulation(cell, sec, i_amp, v_init, tstop, dt, pos_i, pos_v, onset=0, cut_onset=True):
        """
        Runs a NEURON simulation of the cell for the given parameters.

        :param i_amp: Amplitude of the injected current for all times t.
        :type i_amp: array_like
        :param v_init: Initial membrane potential of the cell.
        :type v_init: float
        :param tstop: Duration of a whole run.
        :type tstop: float
        :param dt: Time step.
        :type dt: float
        :param pos_i: Position of the IClamp on the Section (number between 0 and 1).
        :type pos_i: float
        :param pos_v: Position of the recording electrode on the Section (number between 0 and 1).
        :type pos_v: float
        :return: Membrane potential of the cell, time and current amplitude at each time step.
        :rtype: tuple of three ndarrays
        """

        # exchange sec with real Section
        if sec[0] == 'soma':
            section = cell.soma
        elif sec[0] == 'dendrites':
            section = cell.dendrites[sec[1]]
        else:
            raise ValueError('Given section not defined!')

        # time
        t = np.arange(0, tstop + onset + dt, dt)

        # insert an IClamp with the current trace from the experiment
        i_amp = np.concatenate((np.zeros(onset/dt), i_amp))  # during the onset no external current flows
        stim, i_vec, t_vec = section.play_current(pos_i, i_amp, t)

        # record the membrane potential
        v = section.record_v(pos_v)

        # run simulation
        h.v_init = v_init
        h.tstop = tstop + onset
        h.steps_per_ms = 1 / dt  # change steps_per_ms before dt, otherwise dt not changed properly
        h.dt = dt
        h.init()
        h.run()

        if cut_onset:
            return np.array(v)[onset/dt:], t[onset/dt:]
        else:
            return np.array(v), t

def impedance(v, i, dt, f_range):
    """
    Computes the impedance (impedance = fft(v) / fft(i)) for a given range of frequencies.

    :param v: Membrane potential (mV)
    :type v: array
    :param i: Current (nA)
    :type i: array
    :param dt: Time step.
    :type dt: float
    :param f_range: Boundaries of the frequency interval.
    :type f_range: list
    :return: Impedance (MOhm)
    :rtype: array
    """

    # FFT of the membrance potential and the input current
    fft_i = np.fft.fft(i)
    fft_v = np.fft.fft(v)
    freqs = np.fft.fftfreq(v.size, d=dt)

    # sort everything according to the frequencies
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fft_i = fft_i[idx]
    fft_v = fft_v[idx]

    # calculate the impedance
    imp = np.abs(fft_v/fft_i)

    # index with frequency range
    idx1 = np.argmin(np.abs(freqs-f_range[0]))
    idx2 = np.argmin(np.abs(freqs-f_range[1]))

    return imp[idx1:idx2], freqs[idx1:idx2]


def update_diam(cell, variables, variables_new):
    for i, var in enumerate(variables):
        if var[0] == 'L':
            cell.update_attr(variables[i][4], variables[i][5]/(np.pi * variables_new['L']))