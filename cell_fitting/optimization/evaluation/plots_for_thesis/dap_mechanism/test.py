import numpy as np
import matplotlib.pyplot as pl


corr_mat = np.array([[-1.0, 0.0], [0.7, 1.0]])
X, Y = np.meshgrid(np.arange(np.size(corr_mat, 1) +1), np.arange(np.size(corr_mat, 0 ) + 1))  # +1 because otherwise pcolor misses the last row
pl.pcolor(X, Y, np.flipud(corr_mat), vmin=-1, vmax=1, cmap=pl.cm.get_cmap('gray'))
pl.yticks([1, 2],[1, 2])
pl.show()