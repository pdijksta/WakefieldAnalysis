import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt

import myplotstyle as ms

plt.close('all')

with open('./test_profiles.pkl', 'rb') as f:
    all_profiles = pickle.load(f)


ms.figure('Smoothen profile')

subplot = ms.subplot_factory(1, 1)

sp = subplot(1)

bp0 = all_profiles[0]

bp0.plot_standard(sp)

for smoothen in np.array([0.5, 1, 2, 5])*1e-15:
    bp = copy.deepcopy(bp0)
    bp.smoothen(smoothen)
    bp.plot_standard(sp, label='%.1f' % (smoothen*1e15))


sp.legend()



plt.show()

