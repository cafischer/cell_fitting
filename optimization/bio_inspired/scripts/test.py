import scipy.signal
import numpy as np

def get_minima_2d(mat, comp_val, order, mode='clip'):
    def comp(x, y):
        return y-x > comp_val
    min_x = scipy.signal.argrelextrema(mat, comp, axis=0, order=order, mode=mode)
    min_y = scipy.signal.argrelextrema(mat, comp, axis=1, order=order, mode=mode)

    min_x = [tuple(np.array(min_x)[:, j]) for j in range(np.shape(np.array(min_x))[1])]  # transform to tuples with (x, y)
    min_y = [tuple(np.array(min_y)[:, j]) for j in range(np.shape(np.array(min_y))[1])]
    minima = set(min_x).intersection(set(min_y))
    return minima

mat = np.array([[4,3,4,3,0], [5,0.2,3,1,4], [4, 3, 2, 3, 5]])

print get_minima_2d(mat, 0.5, 6, mode='clip')