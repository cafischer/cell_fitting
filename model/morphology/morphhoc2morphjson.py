import re
from json_utils import *
__author__ = 'caro'

# Note: connection point of axon is not extracted from file


f = open('./stellate-garden.hoc','r')

morph_hoc = f.read()
f.close()

# split into soma, dendrite, axon information
geominf_dendrites = re.split('connect dendrites', morph_hoc)  # split parameters according to dendrites
geominf_axon = re.split('axon', geominf_dendrites[-1])[2] # split dendrite and axon parameters
geominf_dendrites[-1] = re.split('axon', geominf_dendrites[-1])[0]  # split dendrite and axon parameters
geominf_soma = geominf_dendrites[0]  # split dendrite and soma parameters
geominf_dendrites = geominf_dendrites[1:]  # split dendrite and soma parameters

# find pt3data of soma
geom = []
pt3d_inf = re.findall("\([-+]?\d*.?\d*,[-+]?\d*\.?\d*,[-+]?\d*\.?\d*,[-+]?\d*\.?\d*\)", geominf_soma)  # find geometry data
for pt3d in pt3d_inf:
    geom.append([float(n) for n in re.findall("[-+]?\d+\.?\d*", pt3d)])  # make them to floats
geomdict_soma = {"geom": geom}  # create structure for a json cell

# find connect statement of dendrites
conn_inf = re.findall('\{[a-z]+[_a-z\[\]\d]*\sconnect\sdendrites\[\d+\]\(\d\),\s\d\\.?\d*\}', morph_hoc)

# find parents of dendrites
parents = []  # parent of the dendrite
parents_n = []  # index of the parent dendrite, None if parent is soma
for inf in conn_inf:
    parents.append((re.match('[a-z]+[_a-z]*', inf[1:])).group(0))
    if re.match('[a-z]+\[(\d*)\]', inf[1:]) is None:
        parents_n.append(None)
    else:
        parents_n.append(int(re.match('[a-z]+\[(\d*)\]', inf[1:]).group(1)))

# find connection point of dendrites
conn_points = []
for inf in conn_inf:
    conn_points.append(float(re.findall('\d\.?\d?', inf[-4:-1])[0]))

# find pt3d data of dendrites
geomdict_dendrites = dict()
for i, inf in enumerate(geominf_dendrites):
    geom = []
    pt3d_inf = re.findall("\([-+]?\d*.?\d*,[-+]?\d*\.?\d*,[-+]?\d*\.?\d*,[-+]?\d*\.?\d*\)", inf)  # find geometry data
    for pt3d in pt3d_inf:
        geom.append([float(n) for n in re.findall("[-+]?\d+\.?\d*", pt3d)])  # make them to floats
    geomdict_dendrites[str(i)] = {"geom": geom, "parent": [parents[i], parents_n[i]],
                 "connection_point": conn_points[i]}  # create structure for a json cell


# find pt3d data of the axon
geomdict_axon = dict()
geom = []
pt3d_inf = re.findall("\([-+]?\d*.?\d*,[-+]?\d*\.?\d*,[-+]?\d*\.?\d*,[-+]?\d*\.?\d*\)", geominf_axon)  # find geometry data
for pt3d in pt3d_inf:
    geom.append([float(n) for n in re.findall("[-+]?\d+\.?\d*", pt3d)])  # make them to floats
geomdict_axon['0'] = {"geom": geom, "parent": ['soma', None],
                 "connection_point": 0.5}  # create structure for a json cell

# save
save_as_json('morph', {'soma': geomdict_soma, 'dendrites': geomdict_dendrites, 'axon_secs': geomdict_axon})






