import numpy as np
from scipy.constants import c
import pickle

import myplotstyle as ms

ms.plt.close('all')

n_slices = 10

with open('./beam_fat_quad.pkl', 'rb') as f:
    beam_fat_quad = pickle.load(f)

with open('./beam_fat_no_quad.pkl', 'rb') as f:
    beam_fat_no_quad = pickle.load(f)

ms.figure('Compare quad no quad')
ms.plt.subplots_adjust(hspace=0.3)
subplot = ms.subplot_factory(2,3)
sp_ctr = 1

sp_slice = subplot(sp_ctr, title='Slice X centroids', xlabel='s [$\mu$m]', ylabel='x [$\mu$m]')
sp_ctr += 1

sp_slice_xp = subplot(sp_ctr, title='Slice Xp centroids', xlabel='s [$\mu$m]', ylabel='xp [$\mu$rad]')
sp_ctr += 1

def calc_slice(beam_s, beam_x, n_slices):
    borders = np.linspace(beam_s.min(), beam_s.max(), n_slices+1)

    mean_s_arr = np.zeros(n_slices)
    mean_x_arr = mean_s_arr.copy()
    size_x_arr = mean_s_arr.copy()

    for n_slice in range(n_slices):
        mask = np.logical_and(beam_s > borders[n_slice], beam_s < borders[n_slice+1])
        mean_x_arr[n_slice] = np.mean(beam_x[mask])
        size_x_arr[n_slice] = np.std(beam_x[mask])
        mean_s_arr[n_slice] = np.mean(beam_s[mask])

    return mean_s_arr, mean_x_arr, size_x_arr


for beam, label in [(beam_fat_quad, 'Quad'), (beam_fat_no_quad, 'No Quad')]:

    for sp, index, title, ax_label in [(sp_slice, 0, 'X', 'x [$\mu$m]'), (sp_slice_xp, 1, 'Xp', 'xp [$\mu$rad]')]:
        ss, xx, xx_err = calc_slice(beam[4,:], beam[index,:], n_slices)
        sp.errorbar(ss*1e6, xx*1e6, yerr=xx_err*1e6, label=label)

        sp_2d = subplot(sp_ctr, grid=False, title=title+' '+label, xlabel='s [$\mu$m]', ylabel=ax_label)
        sp_ctr += 1
        beam_hist, s_edges, x_edges = np.histogram2d(beam[4,:]*c, beam[index,:], bins=(500, 200))
        s_edges *= 1e6
        x_edges *= 1e6
        extent = [s_edges[0], s_edges[-1], x_edges[0], x_edges[-1]]
        sp_2d.imshow(beam_hist.T, aspect='auto', extent=extent, origin='lower')
        hist_s = beam_hist.sum(axis=1)
        diff_x = x_edges.max() - x_edges.min()

        hist_x = beam_hist.sum(axis=0)
        diff_s = s_edges.max() - s_edges.min()
        sp_2d.plot(s_edges[:-1], hist_s/hist_s.max()*diff_x*0.4+x_edges.min(), color='red')
        sp_2d.plot(hist_x/hist_x.max()*diff_s*0.4+s_edges.min(), x_edges[:-1], color='red')

sp_slice.legend()
sp_slice_xp.legend()







ms.plt.show()

