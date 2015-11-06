from __future__ import division
from neuron import nrn, h
import json
import numpy as np
import pylab as pl

__author__ = 'caro'


class Mechanism(object):
    """
    Encapsulates NEURON mechanisms.

    The mechanism is initialized by its name specified as SUFFIX in the .mod file. Additional parameters can be
    delivered that will be set when the Mechanism is inserted into a Section.

    :ivar name: Name of the Mechanism as specified at SUFFIX in the .mod file.
    :type name: str
    :ivar params: Dictionary containing the parameters to be set. Only possible for parameters that are specified
    in the PARAMETER block of the corresponding .mod file.
    :type params: dict

    :Examples:
    leak = Mechanism('pas', {'e': -65, 'g': 0.0002})
    hh = Mechanism('hh')
    """

    def __init__(self, name, params=None):
        """
        Initializes a Mechanism. (But does not insert it yet.)
        """
        self.name = name
        self.params = params

    def insert_into(self, section):
        """
        Inserts a Mechanism into a Section and sets the parameters specified in self.params.

        :param section: Section where the mechanism is inserted.
        :type section: Section
        """
        section.insert(self.name)
        if self.params is not None:
            for name, value in self.params.items():
                for segment in section:
                    mech = getattr(segment, self.name)
                    setattr(mech, name, value)


class Section(nrn.Section):
    """
    Encapsulates NEURON sections.

    :cvar PROXIMAL: Number corresponding to the origin of a Section.
    :type PROXIMAL: float
    :cvar DISTAL: Number corresponding to the end of a Section.
    :type DISTAL: float
    :ivar l: Length (um).
    :type l: float
    :ivar diam: Diameter (um).
    :type diam: float
    :ivar nseg: Number of segments. Specifies the number of internal points at which NEURON computes solutions.
    :type nseg: int
    :ivar ra: Axial resistance (Ohm * cm).
    :type ra: float
    :ivar cm: Membrane capacitance (uF * cm^2)
    :type cm: float
    :ivar mechanisms: Mechanisms (ion channels)
    :type mechanisms: list of Mechanisms
    :ivar parent: Section to which this Section shall be connected.
    :type parent: Section
    :ivar connection_point: Number between 0 and 1 indicating the point at which this Section will be attached on
    the parent Section.
    :type connection_point: float

    :Examples:
    soma = Section(geom={'L':30, 'diam':30}, mechanisms={'hh': {}, 'pas': {}})
    apical = Section(geom={'L':600, 'diam':2}, nseg=5, mechanisms={'pas': {}},
                    parent=soma, connection_point=DISTAL)
    """

    PROXIMAL = 0
    DISTAL = 1

    def __init__(self, geom, nseg=1, Ra=100, cm=1, mechanisms=None, parent=None, connection_point=DISTAL):
        """
        Initializes a Section.
        """
        nrn.Section.__init__(self)  # important for inheritance from NEURON

        # set geometry
        if 'L' in geom and 'diam' in geom:
            self.L = geom['L']
            self.diam = geom['diam']
        else:
            self.set_geom(geom)
        self.nseg = nseg

        # set cable properties
        self.Ra = Ra
        self.cm = cm

        # connect to parent section
        if parent:
            self.connect(parent, connection_point, self.PROXIMAL)

        # add ion channels
        if mechanisms is not None:
            for k, v in mechanisms.iteritems():
                Mechanism(k, v).insert_into(self)

        # add spike_count
        self.spike_count = None  # sustains NEURON reference for recording spikes (see :func Cell.record_spikes)

    def set_geom(self, geom):
        self.push()  # neccessary to access Section in NEURON
        h.pt3dclear()
        for g in geom:
            h.pt3dadd(g[0], g[1], g[2], g[3])
        h.pop_section()  # restore the previously accessed Section

    def record_v(self, pos):
        """
        Records the membrane potential. Values are updated after each NEURON h.run().

        :param: pos: Indicates the position on the Section at which is recorded (number between 0 and 1).
        :type: pos: float
        :return: v: Membrane potential.
        :rtype: v: NEURON Vector
        """
        v = h.Vector()
        v.record(self(pos)._ref_v)
        return v

    def record_spikes(self, pos, threshold=-30):
        """
        Records the spikes of the Cell. Values are updated after each NEURON h.run().

        :param: pos: Indicates the position on the Section at which is recorded (number between 0 and 1).
        :type: pos: float
        :param: threshold: Only spikes above this threshold are counted as spikes.
        :type: threshold: float
        :return: vec: Contains the times where spikes occurred
        :rtype: vec: NEURON vector
        """
        vec = h.Vector()
        self.spike_count = h.APCount(pos, sec=self)  # spike_count assigned to self to keep NEURON reference
        self.spike_count.thresh = threshold
        self.spike_count.record(vec)
        return vec

    def play_current(self, pos, i_amp, t):
        """
        At each time step inject a current equivalent to i_amp at this time step.

        :param: pos: Indicates the position of the IClamp on the Section (number between 0 and 1).
        :type: pos: float
        :param i_amp: Current injected at each time step.
        :type i_amp: array_like
        :param t: Time at each time step.
        :type t: array_like
        :return: IClamp and the current and time vector (NEURON needs the reference).
        :rtype: h.IClamp, h.Vector, h.Vector
        """

        stim = h.IClamp(pos, sec=self)
        stim.delay = 0  # 0 necessary for playing the current into IClamp
        stim.dur = 1e9  # 1e9 necessary for playing the current into IClamp
        i_vec = h.Vector()
        i_vec.from_python(i_amp)
        t_vec = h.Vector()
        t_vec.from_python(t)
        i_vec.play(stim._ref_amp, t_vec) # play current into IClamp (use experimental current trace)
        return stim, i_vec, t_vec


class Cell(object):
    """
    Cell consisting of a soma, a number of dendrites and a number of axon segments.

    The Cell is created from and saved as a .json file. Its format is specified in :param: params in
    :func: Cell.__init__.

    :ivar celsius: Temperature (C).
    :type celsius: float
    :ivar rm: Membrane resistance (Ohm * cm^2).
    :type rm:float
    :ivar soma: Soma (cell body).
    :type soma: Section
    :ivar dendrites: Dendrites.
    :type dendrites: list of Sections
    :ivar axon_secs: Sections of the axon (e.g. axon hillock, axon initial segment, axon).
    :type axon_secs: list of Sections
    """

    def __init__(self, model_dir, mechanism_dir=None):
        """
        Initializes a Cell.

        :param model_dir: Path to the .json file containing the Cells parameters. The .json file has to be composed of
        dictionaries as follows:
        {
        "soma":{"parent", "diam", "ra", "cm", "nseg", "connection_point", "l", "mechanisms":{"name":{"params"}}},
        "dendrites":{"0":{"parent", "diam", "ra", "cm", "nseg", "connection_point", "l",
                    "mechanisms":{"name":{"params"}}}},
        "axon_secs":{"0":{"parent", "diam", "ra", "cm", "nseg", "connection_point", "l",
                    "mechanisms":{"name":{"params"}}}},
        "rm": 50000,
        "celsius": 36,
        "ion": {"0":{"name", "params"}
        }
        whereby the specified fields have to be filled with corresponding values (if not set by default). Dictionaries
        containing "0" can be expanded using "1", "2", etc. followed by a dictionary with the same format as "0".
        :type model_dir: str
        :param mechanism_dir: Specifies the path to the .mod files of the mechanisms (see :func Cell.load_mech)
        :type mechanism_dir: str
        """

        # load .json file
        fr = open(model_dir, 'r')
        params = json.load(fr)

        # assign parameters
        self.params = params

        # load mechanisms (ion channel implementations)
        if mechanism_dir is not None:
            self.load_mech(mechanism_dir)  # must be loaded before insertion of Mechanisms!

        # default parameters
        self.celsius = 36
        self.rm = 10000
        self.soma = Section(geom={'L':20, 'diam':20}, nseg=1, Ra=100, cm=1, mechanisms={'hh': {}}, parent=None)
        self.dendrites = []
        self.axon_secs = []

        # create Cell with given parameters
        self.create(params)

    def create(self, params):
        """
        Creates the cell from params.

        :param params: Cell parameters composed as described in :param model_dir in func: Cell.__init__
        :type params: dict
        """

        # set temperature
        self.celsius = params['celsius']
        h.celsius = self.celsius  # set celsius also in hoc

        # set membrane resistance
        self.rm = params['rm']

        # create sections
        self.soma = Section(**params['soma'])

        self.dendrites = [0] * len(params['dendrites'])
        for i in range(len(params['dendrites'])):
            if params['dendrites'][str(i)]['parent'][0] == 'soma':  # TODO
                params['dendrites'][str(i)]['parent'] = self.soma
            elif params['dendrites'][str(i)]['parent'][0] == 'dendrites':
                params['dendrites'][str(i)]['parent'] = self.dendrites[params['dendrites'][str(i)]['parent'][1]]
            self.dendrites[i] = Section(**params['dendrites'][str(i)])

        self.axon_secs = [0] * len(params['axon_secs'])
        for i in range(len(params['axon_secs'])):
            if params['axon_secs'][str(i)]['parent'][0] == 'soma':  # TODO
                params['axon_secs'][str(i)]['parent'] = self.soma
            elif params['axon_secs'][str(i)]['parent'][0] == 'axon_secs':
                params['axon_secs'][str(i)]['parent'] = self.axon_secs[params['axon_secs'][str(i)]['parent'][1]]

            self.axon_secs[i] = Section(**params['axon_secs'][str(i)])

        # set reversal potentials, insert ions
        for sec in h.allsec():
            for i in range(len(params['ion'])):
                if h.ismembrane(str(params['ion'][str(i)]["name"]), sec=sec):
                    for key, val in params['ion'][str(i)]["params"].iteritems():
                        setattr(sec, key, val)

    def update_attr(self, keys, value):
        """
        Updates the value of an attribute in self.params and recreates the Cell with the new parameters.

        :param keys: List of keys leading to the attribute in self.params.
        :type keys: list of str
        :param value: New value of the attribute.
        :type value: type depends on the attribute
        """
        # update value in self.params
        reduce(lambda dic, key: dic[key], [self.params] + keys[:-1])[keys[-1]] = value  # tracks all changes for saving

        # update Cell
        self.create(self.params)

    def get_attr(self, keys):
        """
        Returns the value of the attribute indexed by keys.

        :param keys: List of keys leading to the attribute in self.params.
        :type keys: list of str
        :return: Value of the accessed attribute.
        :rtype: type depends on the attribute
        """
        value = reduce(lambda dic, key: dic[key], [self.params] + keys[:-1])[keys[-1]]
        return value

    def save_as_json(self, file_dir):
        """
        Saves all parameters of the Cell as .json file from which it can be initialized again.

        :param file_dir: Path under which the Cell parameters shall be stored.
        :type file_dir: str
        """
        if '.json' not in file_dir:
            file_dir += '.json'
        fw = open(file_dir, 'w')
        json.dump(self.params, fw, indent=4)

    @staticmethod
    def load_mech(mechanism_dir):
        """
        Loads the mechanisms (ion channel kinetics) written in NMODL. The same mechanisms cannot be loaded twice.

        :param mechanism_dir: Specifies the path to the .mod files. Mod files have to be compiled in the same folder
        using nrnivmodl.
        :type mechanism_dir: str
        """
        # load membrane mechanisms
        h.nrn_load_dll(mechanism_dir + '/i686/.libs/libnrnmech.so')


#######################################################################################################################


def test_mechanism_insertion():
    print "Test Mechanism insertion: "

    # create a Mechanism
    params = {'gnabar': 0.2}

    # create a Section (automatically inserts the Mechanims)
    sec = Section(geom={'L':30, 'diam':30}, mechanisms={'hh': params})

    # print the variables of the Mechanism in the Section
    if sec.gnabar_hh == params['gnabar']:
        print "Correct!"
    else:
        print "Wrong!"


def test_record():
    print "Test record membrane potential and spikes from a Section: "
    print "See figure."

    import pylab as pl
    import numpy as np
    h.load_file("stdrun.hoc")

    # create a Section
    sec = Section(geom={'L':15, 'diam':15}, mechanisms={'hh':{}, 'pas':{}})

    # record
    pos = 0.5
    v = sec.record_v(pos)
    t = h.Vector()
    t.record(h._ref_t)
    spikes = sec.record_spikes(pos)

    # stimulate
    stim = h.IClamp(pos, sec=sec)
    stim.delay = 50
    stim.dur = 400
    stim.amp = 0.3

    # run
    h.dt = 0.025
    h.tstop = 500
    h.run()

    # plot
    pl.plot(t, v, 'k', label='v')
    pl.axhline(min(v)-10, color='k')
    pl.plot(spikes, np.ones(len(spikes))*min(v)-10, 'or', label='spikes')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.title('Regular spiking')
    pl.show()


def test_cell_creation():
    print "Test the creation of a Cell: "

    # create Cell
    cell = Cell('../demo/demo_cell.json')

    # test if attribute from StellateCell.json were set
    print "Attribute set: "
    if cell.dendrites[0].L == cell.params['dendrites']['0']['geom']['L']:
        print "Yes!"
    else:
        print "No!"

    # change attribute and test again
    print "Update of attribute: "
    val = 222
    cell.update_attr(['dendrites', '0', 'geom', 'L'], val)
    if cell.dendrites[0].L == val and cell.params['dendrites']['0']['geom']['L'] == val:
        print "Correct!"
    else:
        print "Wrong!"

    # save new Cell parameters to a .json file
    print "Cell saved and retrieved: "
    cell.save_as_json('../demo/demo_cell_new')
    cell_new = Cell('../demo/demo_cell_new.json')
    if cell_new.dendrites[0].L == val:
        print "Correct!"
    else:
        print "Wrong!"


def test_compare_to_hoc_cell():
    print "Compare the membrane potential of the cell created from a .json file and the cell created from a .hoc file."
    print "See figure. Both should be the same!"

    # create Cell from .json file
    cell_json = Cell('../demo/demo_cell2.json')

    # create NEURON cell from .hoc file
    h.xopen("../demo/demo_cell2.hoc")

    # record membrane potential
    v_json = cell_json.soma.record_v(0.5)
    v_hoc = h.Vector()
    v_hoc.record(h.soma(0.5)._ref_v)

    # inject current
    stim_json = h.IClamp(0.5, sec=cell_json.soma)
    stim_hoc = h.IClamp(0.5, sec=h.soma)
    stim_json.delay = 50
    stim_json.dur = 200
    stim_json.amp = 0.1
    stim_hoc.delay = 50
    stim_hoc.dur = 200
    stim_hoc.amp = 0.1

    i_amp = h.Vector()
    i_amp.record(stim_json._ref_i) # record the current amplitude (to check)

    # run simulation
    h.tstop = 300
    h.init()
    h.run()

    t = np.arange(0, h.tstop+h.dt, h.dt)

    # plot the results
    f, (ax1, ax2) = pl.subplots(2, 1, sharex=True)
    ax1.plot(t, np.array(v_json), 'b', label='Cell_json')
    ax1.plot(t, np.array(v_hoc), 'r', label='Cell_hoc')
    ax1.set_ylabel('Membrane potential (mV)')
    ax1.legend()
    ax2.plot(t, np.array(i_amp), 'k')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Current (nA)')
    pl.show()


def test_morph():
    from mpl_toolkits.mplot3d import Axes3D

    # cell with complex morphology
    cell = Cell('./cells/StellateCell_full.json')

    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot dendrites
    for sec_i, sec in enumerate([cell.soma]+cell.dendrites+cell.axon_secs):
        sec.push()
        X = [0]*int(h.n3d())
        Y = [0]*int(h.n3d())
        Z = [0]*int(h.n3d())
        d = [0]*int(h.n3d())
        for i in range(int(h.n3d())):
            X[i] = h.x3d(i)
            Y[i] = h.y3d(i)
            Z[i] = h.z3d(i)
            d[i] = h.diam3d(i)
        if sec_i>len(cell.dendrites)+1:
            ax.plot(X, Y, Z, 'r', linewidth=d[0])
        else:
            ax.plot(X, Y, Z, 'k', linewidth=d[0])
        h.pop_section()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-200,200])
    ax.set_ylim([-200,200])
    pl.show()
    #pl.savefig('morph.svg')


if __name__ == "__main__":

    #test_mechanism_insertion()

    #test_record()

    #test_cell_creation()

    #test_compare_to_hoc_cell()

    test_morph()


# TODO: flag for geometry
# TODO: connect statements for morph, better way?