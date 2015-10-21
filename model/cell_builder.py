from neuron import nrn, h
import json

__author__ = 'caro'


class Mechanism(object):
    """
    Encapsulates NEURON mechanisms.

    The mechanism is initialized by its name specified as SUFFIX in the .mod file. Additional parameters can be
    delivered that will be set when the Mechanism is inserted into a Section.

    :ivar name: Name of the Mechanism as specified at SUFFIX in the .mod file.
    :type name: String
    :ivar params: Dictionary containing the parameters to be set. Only possible for parameters that are specified
    in the PARAMETER block of the corresponding .mod file.
    :type params: Dict

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
    :type PROXIMAL: Float
    :cvar DISTAL: Number corresponding to the end of a Section.
    :type DISTAL: Float
    :ivar l: Length (um).
    :type l: Float
    :ivar diam: Diameter (um).
    :type diam: FLoat
    :ivar nseg: Number of segments. Specifies the number of internal points at which NEURON computes solutions.
    :type nseg: Int
    :ivar ra: Axial resistance (Ohm * cm).
    :type ra: Float
    :ivar cm: Membrane capacitance (uF * cm^2)
    :type cm: Float
    :ivar mechanisms: Mechanisms (ion channels)
    :type mechanisms: List of Mechanisms
    :ivar parent: Section to which this Section shall be connected.
    :type parent: Section
    :ivar connection_point: Number between 0 and 1 indicating the point at which this Section will be attached on
    the parent Section.
    :type connection_point: Float

    :Examples:
    soma = Section(l=30, diam=30, mechanisms=[Mechanism('hh'), Mechanism('pas')])
    apical = Section(l=600, diam=2, nseg=5, mechanisms=[Mechanism('pas')],
                    parent=soma, connection_point=DISTAL)
    """

    PROXIMAL = 0
    DISTAL = 1

    def __init__(self, L, diam, nseg=1, Ra=100, cm=1, mechanisms=None, parent=None, connection_point=DISTAL):
        """
        Initializes a Section.
        """
        nrn.Section.__init__(self)  # important for inheritance from NEURON

        # set geometry
        self.L = L # TODO L change everywhere , NEURON depended on same names?
        self.diam = diam
        self.nseg = nseg

        # set cable properties
        self.Ra = Ra
        self.cm = cm

        # connect to parent section
        if parent:
            self.connect(parent, connection_point, self.PROXIMAL)

        # add ion channels
        if mechanisms is not None:
            for mechanism in mechanisms:
                mechanism.insert_into(self)

        # add spike_count
        self.spike_count = None  # sustains NEURON reference for recording spikes (see :func Cell.record_spikes)

    def record_v(self, pos):
        """
        Records the membrane potential. Values are updated after each NEURON h.run().

        :param: pos: Indicates the position on the Section at which is recorded (number between 0 and 1).
        :type: pos: Float
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
        :type: pos: Float
        :param: threshold: Only spikes above this threshold are counted as spikes.
        :type: threshold: Float
        :return: vec: Contains the times where spikes occurred
        :rtype: vec: NEURON vector
        """
        vec = h.Vector()
        self.spike_count = h.APCount(pos, sec=self)  # spike_count assigned to self to keep NEURON reference
        self.spike_count.thresh = threshold
        self.spike_count.record(vec)
        return vec


class Cell(object):
    """
    Cell consisting of a soma, a number of dendrites and a number of axon segments.

    The Cell is created from and saved as a .json file. Its format is specified in :param: params in
    :func: Cell.__init__.

    :ivar celsius: Temperature (C).
    :type celsius: Float
    :ivar rm: Membrane resistance (Ohm * cm^2).
    :type rm:Float
    :ivar soma: Soma (cell body).
    :type soma: Section
    :ivar dendrites: Dendrites.
    :type dendrites: List of Sections
    :ivar axon_secs: Sections of the axon (e.g. axon hillock, axon initial segment, axon).
    :type axon_secs: List of Sections
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
        :type model_dir: String
        :param mechanism_dir: Specifies the path to the .mod files of the mechanisms (see :func Cell.load_mech)
        :type mechanism_dir: String
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
        self.soma = Section(L=20, diam=20, nseg=1, Ra=100, cm=1, mechanisms=[Mechanism('hh')], parent=None)
        self.dendrites = []
        self.axon_secs = []

        # create Cell with given parameters
        self.create(params)

    def create(self, params):
        """
        Creates the cell from params.

        :param params: Cell parameters composed as described in :param model_dir in func: Cell.__init__
        :type params: Dict
        """

        # set temperature
        self.celsius = params['celsius']
        h.celsius = self.celsius  # set celsius also in hoc

        # set membrane resistance
        self.rm = params['rm']

        # create sections
        self.soma = Section(L=params['soma']['L'], diam=params['soma']['diam'], nseg=params['soma']['nseg'],
                            Ra=params['soma']['Ra'], cm=params['soma']['cm'],
                            mechanisms=[Mechanism(k, v)
                                        for k, v in params['soma']['mechanisms'].iteritems()],
                            parent=params['soma']['parent'], connection_point=params['soma']['connection_point'])

        self.dendrites = [0] * len(params['dendrites'])
        for i in range(len(params['dendrites'])):
            self.dendrites[i] = Section(L=params['dendrites'][str(i)]['L'], diam=params['dendrites'][str(i)]['diam'],
                                        nseg=params['dendrites'][str(i)]['nseg'], Ra=params['dendrites'][str(i)]['Ra'],
                                        cm=params['dendrites'][str(i)]['cm'],
                                        mechanisms=[Mechanism(k, v)
                                                    for k, v in params['dendrites'][str(i)]['mechanisms'].iteritems()],
                                        parent=params['dendrites'][str(i)]['parent'],
                                        connection_point=params['dendrites'][str(i)]['connection_point'])

        self.axon_secs = [0] * len(params['axon_secs'])
        for i in range(len(params['axon_secs'])):
            self.axon_secs[i] = Section(L=params['axon_secs'][str(i)]['L'], diam=params['axon_secs'][str(i)]['diam'],
                                        nseg=params['axon_secs'][str(i)]['nseg'], Ra=params['axon_secs'][str(i)]['Ra'],
                                        cm=params['axon_secs'][str(i)]['cm'],
                                        mechanisms=[Mechanism(k, v)
                                                    for k, v in params['axon_secs'][str(i)]['mechanisms'].iteritems()],
                                        parent=params['axon_secs'][str(i)]['parent'],
                                        connection_point=params['axon_secs'][str(i)]['connection_point'])

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
        :type keys: List
        :param value: New value of the attribute.
        :type value: Type depends on the attribute to be changed.
        """
        # update value in self.params
        reduce(lambda dic, key: dic[key], [self.params] + keys[:-1])[keys[-1]] = value  # tracks all changes for saving

        # update Cell
        self.create(self.params)

    def save_as_json(self, filename):
        """
        Saves all parameters of the Cell as .json file from which it can be initialized again.

        :param filename: Path under which the Cell parameters shall be stored.
        :type filename: String
        """
        if '.json' not in filename:
            filename += '.json'
        fw = open(filename, 'w')
        json.dump(self.params, fw, indent=4)

    @staticmethod
    def load_mech(mechanism_dir):
        """
        Loads the mechanisms (ion channel kinetics) written in NMODL. The same mechanisms cannot be loaded twice.

        :param mechanism_dir: Specifies the path to the .mod files. Mod files have to be compiled in the same folder
        using nrnivmodl.
        :type mechanism_dir: String
        """
        # load membrane mechanisms
        h.nrn_load_dll(mechanism_dir + '/i686/.libs/libnrnmech.so')


#######################################################################################################################

def test_mechanism_insertion():

    # create a Mechanism
    params = {'gnabar': 0.2}
    m = Mechanism('hh', params)

    # create a Section (automatically inserts the Mechanims)
    sec = Section(l=30, diam=30, mechanisms=[m])

    # print the variables of the Mechanism in the Section
    if sec.gnabar_hh == params['gnabar']:
        print "Mechanism inserted and parameter correct!"
    else:
        print "Mechanism not correctly inserted!"


def test_record():

    import pylab as pl
    import numpy as np
    h.load_file("stdrun.hoc")

    # create a Section
    sec = Section(l=30, diam=30, mechanisms=[Mechanism('hh'), Mechanism('pas')])

    # record
    pos = 0.5
    v = sec.record_v(pos)
    t = h.Vector()
    t.record(h._ref_t)
    spikes = sec.record_spikes(pos)

    # stimulate
    stim = h.IClamp(pos, sec=sec)
    stim.delay = 50
    stim.dur = 1000
    stim.amp = 1

    # run
    h.dt = 0.025
    h.tstop = 1100
    h.run()

    # plot
    pl.plot(t, v, 'k', label='v')
    pl.axhline(min(v)-10, color='k')
    pl.plot(spikes, np.ones(len(spikes))*min(v)-10, 'or', label='spikes')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.show()


def test_cell_creation():

    # create Cell
    cell = Cell('demo/demo_cell.json')

    # test if attribute from StellateCell.json were set
    if cell.dendrites[0].l == cell.params['dendrites']['0']['l']:
        print "Attribute is correct!"
    else:
        print "Attribute is not correct!"

    # change attribute and test again
    val = 222
    cell.update_attr(['dendrites', '0', 'l'], val)
    if cell.dendrites[0].l == val and cell.params['dendrites']['0']['l'] == val:
        print "Successful update of the attribute!"
    else:
        print "Attribute not correctly updated!"

    # save new Cell parameters to a .json file
    cell.save_as_json('demo/demo_cell_new')
    cell_new = Cell('demo/demo_cell_new.json')
    if cell_new.dendrites[0].l == val:
        print "Parameters saved and retrieved correctly from .json file."


if __name__ == "__main__":

    test_mechanism_insertion()

    test_record()

    test_cell_creation()