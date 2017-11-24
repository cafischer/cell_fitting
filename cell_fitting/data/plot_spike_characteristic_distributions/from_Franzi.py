import pandas as pd
import matplotlib.pyplot as pl
import numpy as np


data_dir = '/home/cf/Phd/DAP-Project/cell_data/division_rat_gerbil/Data+Immuno_Rat.csv'
rat_data = pd.read_csv(data_dir, header=1)

DAP_deflection = rat_data.DAP_defl.values
pl.figure()
pl.title('DAP deflection')
pl.hist(DAP_deflection[~np.isnan(DAP_deflection)], 100)
pl.show()

DAP_width = rat_data.DAP_width.values
pl.figure()
pl.title('DAP width')
pl.hist(DAP_width[~np.isnan(DAP_width)], 100)
pl.show()