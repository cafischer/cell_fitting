import numpy as np
import matplotlib.pyplot as pl

t = np.arange(0, 10, 0.01)
pl.figure()
pl.plot(t, np.exp(-t), 'k')
pl.plot(t, np.exp(-t/2), 'g')
pl.plot(t, np.exp(-t/3), 'r')
pl.plot(t, np.exp(-t/4), 'g')
pl.show()