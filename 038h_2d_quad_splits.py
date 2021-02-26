import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.constants import c
import scipy.interpolate

import myplotstyle as ms
import wf_model

len_s = int(2e3)
len_x = int(1e3)

plt.close('all')

filename = './investigate_streaking.pkl'

with open(filename, 'rb') as f:
    d = pickle.load(f)

def drift(L):
    return np.array([
        [1, L, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, L, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],], float)



beam0 = drift(-0.5) @ d['beam_before_s2']
bp = d['profile']
semigap = d['gap']/2.
beam_offset = d['beam_offset']
do_quad_wake = True

n_splits = 5
n_slices = 10

beam_offset = 4.7e-3
len_streaker = 1
energy_eV = bp.energy_eV
delta_l = len_streaker/n_splits/2

beam = beam0.copy()

centroids = [beam[0,:].mean()]
centroids_xp = [beam[1,:].mean()]
s_list = [0]
s = 0


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


for n_split in range(n_splits):


    fig = ms.figure('Test 2d interpolation step %i' % n_split)
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    ss = np.linspace(0, (beam[4,:].max() - beam[4,:].min())*c*1.01, len_s)
    xx = np.linspace(beam[0,:].min(), beam[0,:].max(), len_x)+beam_offset

    distance = semigap - beam_offset


    sp = sp0 = subplot(sp_ctr, title='Single particle wake', xlabel='s [$\mu$m]', ylabel='Wake [kV/pC/m]')
    sp_ctr += 1

    for index in [0, len_x//2, -1]:
        z = wf_model.wxd(ss, semigap, xx[index])
        distance = semigap - xx[index]
        sp.plot(ss*1e6, z*1e-15, label='%.1f' % (distance*1e6))

    probe_s = beam[4,:] * c
    probe_x = beam[0,:]

    wake2d_dict = wf_model.wf2d(probe_s, probe_x, beam_offset, semigap, bp.charge, wf_model.wxd)
    beam_hist = wake2d_dict['beam_hist']
    s_edges = wake2d_dict['s_bins']
    x_edges = wake2d_dict['x_bins']



    sp = sp1 = subplot(sp_ctr, title='Dipole wake potential', xlabel='s [$\mu$m]', ylabel='Wake [MV/m]')
    sp_ctr += 1

    sp = subplot(sp_ctr, title='Current hist', xlabel='s', ylabel='x', scix=True, sciy=True, grid=False)
    sp_ctr += 1

    sp.imshow(np.abs(beam_hist.T), aspect='auto', extent=[s_edges[0], s_edges[-1], x_edges[0], x_edges[-1]], origin='lower')

    wake_from2d = wake2d_dict['wake']
    wake_s = wake2d_dict['wake_s']

    hist1d, _ = np.histogram(probe_s, s_edges)
    hist1d = hist1d / hist1d.sum() * bp.charge
    abs_hist1d = np.abs(hist1d)
    sp.plot(s_edges[:-1], abs_hist1d/abs_hist1d.max()*(x_edges.max()-x_edges.min())*0.3+x_edges.min(), color='red')
    wf_calc = wf_model.WakeFieldCalculator(s_edges, hist1d, bp.energy_eV, 1)
    spw = wf_model.wxd(wf_calc.xx, semigap, beam_offset)
    convoluted_wake = wf_calc.wake_potential(spw)

    sp1.plot(wf_calc.xx*1e6, convoluted_wake/1e6, label='Thin beam')
    sp1.plot(s_edges*1e6, wake_from2d/1e6, label='Large beam')

    sp0.legend(title='Distance from jaw [$\mu$m]')
    sp1.legend()

    sp = subplot(sp_ctr, title='Single particle wake', xlabel='s', ylabel='x', scix=True, sciy=True, grid=False)
    sp_ctr += 1

    out = sp.imshow(np.abs(wake2d_dict['spw2d']), aspect='auto', extent=[s_edges[0], s_edges[-1], x_edges[0], x_edges[-1]], origin='lower')
    fig.colorbar(out, ax=sp, extend='both')

    wake_effect = np.interp(probe_s, wake_s, wake_from2d) / energy_eV / n_splits * np.sign(bp.charge)

    if do_quad_wake:
        wake2d_dict_quad = wf_model.wf2d_quad(probe_s, probe_x, beam_offset, semigap, bp.charge, wf_model.wxq)
        quad_s = wake2d_dict_quad['s_bins']
        quad_x = wake2d_dict_quad['x_bins']
        quad_wake = wake2d_dict_quad['wake'] / energy_eV / n_splits * np.sign(bp.charge)
        try:
            quad_effect = np.zeros_like(beam[0,:])
            interp_quad = scipy.interpolate.interp2d(quad_s, quad_x, quad_wake.T)
            #import time
            #time0 = time.time()
            for n, (s_, x) in enumerate(zip(probe_s, beam[0,:]+beam_offset)):
                quad_effect[n] = interp_quad(s_, x)
            #print(time.time() - time0, 'seconds')
            #import pdb; pdb.set_trace()
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
    else:
        quad_effect = 0


    if n_split == 0:
        wake_effect0 = wake_effect * n_splits

    beam = drift(delta_l) @ beam
    s += delta_l
    s_list.append(s)

    centroids.append(beam[0,:].mean())
    centroids_xp.append(beam[1,:].mean())

    beam[1,:] += wake_effect + quad_effect
    #import pdb; pdb.set_trace()


    centroids.append(beam[0,:].mean())
    centroids_xp.append(beam[1,:].mean())
    s_list.append(s)


    beam = drift(delta_l) @ beam

    s += delta_l
    s_list.append(s)
    centroids.append(beam[0,:].mean())
    centroids_xp.append(beam[1,:].mean())

beam_fat = beam

fig = ms.figure('Overview')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp = subplot(sp_ctr, title='Centroid x trajectory', xlabel='z [m]', ylabel='$\Delta$ x [um]')
sp_ctr += 1
sp_xp = subplot(sp_ctr, title='Centroid xp trajectory', xlabel='z [m]', ylabel='xp [urad]')
sp_ctr += 1

sp_slice = subplot(sp_ctr, title='Slice X centroids', xlabel='s [$\mu$m]', ylabel='x [$\mu$m]')
sp_ctr += 1

sp_slice_xp = subplot(sp_ctr, title='Slice Xp centroids', xlabel='s [$\mu$m]', ylabel='xp [$\mu$rad]')
sp_ctr += 1

centroids = np.array(centroids)
centroids_xp = np.array(centroids_xp)
s_list = np.array(s_list)

sp.plot(s_list, centroids*1e6, label='%.1f' % (centroids[-1]*1e6))
sp_xp.plot(s_list, centroids_xp*1e6, label='%.1f' % (centroids_xp[-1]*1e6))

beam = d['beam_before_s2'].copy()


centroids = [beam[0,:].mean()]
centroids_xp = [beam[1,:].mean()]
s_list = [0]
s = 0

beam = drift(0.5) @ beam

s += 0.5
s_list.append(s)
centroids.append(beam[0,:].mean())
centroids_xp.append(beam[1,:].mean())

beam[1,:] += wake_effect0

s_list.append(s)
centroids.append(beam[0,:].mean())
centroids_xp.append(beam[1,:].mean())

beam = drift(0.5) @ beam
beam_thin = beam

s += 0.5
s_list.append(s)
centroids.append(beam[0,:].mean())
centroids_xp.append(beam[1,:].mean())


centroids = np.array(centroids)
centroids_xp = np.array(centroids_xp)
s_list = np.array(s_list)

sp.plot(s_list, centroids*1e6, label='%.1f' % (centroids[-1]*1e6))
sp_xp.plot(s_list, centroids_xp*1e6, label='%.1f' % (centroids_xp[-1]*1e6))

for beam, label in [(beam_thin, 'Thin'), (beam_fat, 'Fat')]:
    for sp, index in [(sp_slice, 0), (sp_slice_xp, 1)]:
        ss, xx, xx_err = calc_slice(beam[4,:], beam[index,:], n_slices)
        sp.errorbar(ss*1e6, xx*1e6, yerr=xx_err*1e6, label=label)


sp.legend()
sp_xp.legend()
sp_slice.legend()
sp_slice_xp.legend()


plt.show()

