import matplotlib.pyplot as plt
import numpy as np
import pickle

import myplotstyle as ms

plt.close('all')

filename = './investigate_streaking.pkl'

with open(filename, 'rb') as f:
    d = pickle.load(f)

beam0 = d['beam_before_s2']
delta_xp0 = d['delta_xp']

def drift(L):
    return np.array([
        [1, L, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, L, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],], float)

ms.figure('Split streaker')

subplot = ms.subplot_factory(2, 2)

sp = subplot(1, title='Centroid x trajectory', xlabel='s [m]', ylabel='$\Delta$ x [um]')
sp_xp = subplot(2, title='Centroid xp trajectory', xlabel='s [m]', ylabel='xp [urad]')

len_streaker = 1
distance0 = d['gap']/2 - d['beam_offset']

for n_splits in (1, 5, 10):

    beam = drift(-0.5) @ beam0

    centroids = [beam[0,:].mean()]
    centroids_xp = [beam[1,:].mean()]
    s_list = [0]

    s = 0
    delta_l = len_streaker/n_splits/2

    for n_split in range(n_splits):

        beam = drift(delta_l) @ beam
        s += delta_l
        s_list.append(s)
        centroids.append(beam[0,:].mean())
        centroids_xp.append(beam[1,:].mean())

        delta_xp = delta_xp0 * distance0**2.7 / (distance0-centroids[-1])**2.7
        beam[1,:] += delta_xp / n_splits

        beam = drift(1/n_splits/2) @ beam
        s += delta_l
        s_list.append(s)
        centroids.append(beam[0,:].mean())
        centroids_xp.append(beam[1,:].mean())

    centroids = np.array(centroids)
    centroids_xp = np.array(centroids_xp)
    s_list = np.array(s_list)

    sp.plot(s_list, centroids*1e6, label='%i' % n_splits)
    sp_xp.plot(s_list, centroids_xp*1e6, label='%i' % n_splits)

sp.legend(title='Number of splits')


plt.show()

