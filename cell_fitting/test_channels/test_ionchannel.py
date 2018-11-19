from nrn_wrapper import *
import matplotlib.pyplot as pl

__author__ = 'caro'


def voltage_steps(sec, amps, durs, v_steps, stepamp, pos, dt):
    """
    Inserts a voltage clamp into the cell with the possiblity for three different amplitudes at three different
    durations. The amplitude indicated by stepamp will be used to step through v_steps.

    :param sec: Section of a cell from which shall be recorded.
    :type sec: Section
    :param amps: Three different amplitudes of the VClamp.
    :type amps: list (3 Elements)
    :param durs: Duration of the three different amplitudes.
    :type durs: list (3 Elements)
    :param v_steps: Potential of the different voltage steps.
    :type v_steps: list
    :param stepamp: Number of the amplitude which will be stepped.
    :type stepamp: int
    :param pos: Position at which the VClamp will be inserted.
    :type pos: float
    :param dt: Size of the time step of the simulation.
    :type dt: float
    :return: Current traces for each voltage step and the time axis.
    :rtype: list, array
    """
    # time
    tstop = np.sum(durs)
    t = np.arange(0, tstop+dt, dt)

    # VClamp
    clamp = h.SEClamp(pos, sec=sec)
    clamp.dur1 = durs[0]
    clamp.dur2 = durs[1]
    clamp.dur3 = durs[2]
    clamp.amp1 = amps[0]
    clamp.amp2 = amps[1]
    clamp.amp3 = amps[2]
    clamp.rs = 1e-9  # should be very small
    i_clamp = h.Vector()
    i_clamp.record(clamp._ref_i)

    # run simulation
    i_steps = []
    for v_step in v_steps:
        setattr(clamp, 'amp' + str(stepamp), v_step)
        h.tstop = tstop + dt
        h.steps_per_ms = 1.0/dt
        h.dt = dt
        h.v_init = amps[0]
        h.run()
        i_steps.append(np.array(i_clamp)[1:])  # at first point not initialized yet

    return i_steps, t


def current_subtraction(sec, channel, celsius, amps, durs, v_steps, stepamp, pos, dt):
    """
    Conducts two voltage step experiments: one with the conductance of the respective channel turned on, one with the
    conductance turned of. Then the measured current traces are subtracted to isolate the current of the channel.

    :param sec: Section of a cell from which shall be recorded.
    :type sec: Section
    :param channel: Reference to the channel in the Section.
    :type channel: Section.channel
    :param gbars: List with the names of the conductances that shall be set to 0.
    :type gbars: list
    :param celsius: Temperature of the experiment.
    :type celsius: float
    :param amps: Three different amplitudes of the VClamp.
    :type amps: list (3 Elements)
    :param durs: Duration of the three different amplitudes.
    :type durs: list (3 Elements)
    :param v_steps: Potential of the different voltage steps.
    :type v_steps: array_like
    :param stepamp: Number of the amplitude which will be stepped.
    :type stepamp: int
    :param pos: Position at which the VClamp will be inserted.
    :type pos: float
    :param dt: Size of the time step of the simulation.
    :type dt: float
    :return: Isolated current of the channel for each voltage step and the time axis.
    :rtype: list, array
    """

    i_steps = []

    # create cell
    h.celsius = celsius
    i_steps_control, t = voltage_steps(sec, amps, durs, v_steps, stepamp, pos, dt)
    gbar_channel = getattr(channel, 'gbar', 0.0)
    setattr(channel, 'gbar', 0.0)
    i_steps_blockade, t = voltage_steps(sec, amps, durs, v_steps, stepamp, pos, dt)
    setattr(channel, 'gbar', gbar_channel)

    for i, v_step in enumerate(v_steps):
        i_steps.append(i_steps_control[i] - i_steps_blockade[i])

    return i_steps, t


def spike(sec, dur=0.5, delay=2, amp=1, tstop=6, dt=0.025, v_init=-65):
        """
        Elicits a spike and returns the measured membrane potential and time axis.

        :param sec: Section to record from.
        :type sec: Section
        :param dur: Duration of the stimulation by the IClamp.
        :type dur: float
        :param delay: Start of the IClamp stimulation.
        :type delay: float
        :param amp: Amplitude of the injected current.
        :type amp: float
        :param tstop: Duration of the simulation.
        :type tstop: float
        :param dt: Time step of the simulation.
        :type dt: float
        :param v_init: Membrane potential of the Section at the start of the simulation.
        :type v_init: float
        :return: Membrane potential and time axis.
        :rtype: array, array
        """

        # time
        t = np.arange(0, tstop+dt, dt)

        # record v
        v = sec.record_v(0.5)

        # IClamp
        clamp = h.IClamp(0,5, sec=sec)
        clamp.delay = delay
        clamp.dur = dur
        clamp.amp = amp

        h.tstop = tstop
        h.dt = dt
        h.v_init = v_init
        h.init()
        h.run()

        return np.array(v), t


def plot_i_steps(i_steps, v_steps, t):
    """
    Plots the current traces from all coltage steps.

    :param i_steps: List containing the current traces for each voltage step.
    :type i_steps: list
    :param v_steps: Potential of the different voltage steps.
    :type v_steps: array_like
    :param t: Time axis
    :type t: array
    """

    pl.figure()
    for i, v_step in enumerate(v_steps):
        pl.plot(t, i_steps[i], label=str(v_step))
    pl.legend(loc='lower right', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Current (nA)', fontsize=16)
    pl.tight_layout()
    pl.show()


def plot_i_steps_on_ax(ax, i_steps, v_steps, t):
    """
    Plots the current traces from all coltage steps.

    :param i_steps: List containing the current traces for each voltage step.
    :type i_steps: list
    :param v_steps: Potential of the different voltage steps.
    :type v_steps: array_like
    :param t: Time axis
    :type t: array
    """
    cmap = pl.cm.get_cmap('plasma')
    for i, v_step in enumerate(v_steps):
        ax.plot(t, i_steps[i], label='%i (mV)' % v_step, color=cmap(float(i) / len(v_steps)))
    ax.legend(loc='lower right')
    ax.set_xlabel('Time (ms)')
    #ax.set_ylabel('Current (nA)')
