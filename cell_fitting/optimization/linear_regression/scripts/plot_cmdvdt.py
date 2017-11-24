import matplotlib.pyplot as pl
import pandas as pd

from cell_fitting.optimization.helpers import *

data_dir = '../../../data/2015_08_26b/raw/simulate_rampIV/3.0(nA).csv'
data = pd.read_csv(data_dir)
L = 50  # um
diam = 100  # um
cm = 1  # uF/cm**2

t = np.array(data.t.values)
dt = t[1]  # ms
v_exp = np.array(data.v.values)  # mV
dvdt = np.concatenate((np.array([(v_exp[1] - v_exp[0]) / dt]), np.diff(v_exp) / dt))  # V

# convert units
cell_area = get_cellarea(convert_from_unit('u', L),
                         convert_from_unit('u', diam))  # m**2
Cm = convert_from_unit('c', cm) * cell_area  # F
i_inj = convert_from_unit('n', np.array(data.i.values))  # A

lhs = dvdt * Cm - i_inj  # A

print convert_from_unit('T', Cm)  # pF

pl.figure()
pl.plot(t, lhs, 'k')
pl.xlim(9.5, 14)
pl.show()