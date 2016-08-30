import matplotlib.pyplot as pl
import numpy as np

dict_variable_names = {'a': 1, 'b': 2, 'c': 3}
range_names = range(3)

variable_names_min = ['b', 'a', 'a']
values_min = np.array([dict_variable_names[vn] for vn in variable_names_min])
variable_names_max = ['b', 'c', 'c', 'c']
values_max = np.array([dict_variable_names[vn] for vn in variable_names_max])

h_min = np.histogram(values_min, bins=3)[0]
h_max = np.histogram(values_max, bins=3)[0]

w = 0.3
pl.bar(np.array(range_names)-w/2, h_min, width=w, color='b', align='center')
pl.bar(np.array(range_names)+w/2, h_max, width=w, color='r', align='center')
pl.tight_layout()
pl.show()