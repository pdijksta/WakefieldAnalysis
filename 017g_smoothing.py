import matplotlib.pyplot as plt
import numpy as np
import pickle
import tracking

import myplotstyle as ms

plt.close('all')

with open('./example_screen.pkl', 'rb') as f:
    screen_old = pickle.load(f)

ms.figure('Investigate Smoothing and binning')

subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp = subplot(sp_ctr, title='Histtograms')
sp_ctr += 1


sp2 = subplot(sp_ctr, title='Histograms smooth')
sp_ctr += 1

xx = screen_old.real_x

for n_bins in (150, 300, 450):
    hist, bin_edges = np.histogram(xx, bins=n_bins, density=True)
    screen = tracking.ScreenDistribution(bin_edges[1:], hist)
    screen.smoothen(30e-6)
    sp.plot(bin_edges[1:], hist, label=n_bins)
    sp2.plot(screen.x, screen.intensity, label=n_bins)



plt.show()




