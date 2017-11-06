import numpy as np

# 2014_11_27
# dorsal: b, d, e, f, g, h, i
# ventral: j, k, l, m, o, p

# resonance freq: ~8 dorsal - ~3 ventral
res_dorsal = [6.5, np.nan, 6.2, 5.3, 6, 5.8, 6.2]
res_ventral = [5.2, 5.2, 5.8, 6.5, 4.3, 3.9]

print np.nanmean(res_dorsal)
print np.nanmean(res_ventral)