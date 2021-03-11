import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.constants import c

import myplotstyle as ms
import wf_model

len_s = int(2e3)
len_x = int(1e3)

plt.close('all')

filename = './investigate_streaking.pkl'

with open(filename, 'rb') as f:
    d = pickle.load(f)

beam0 = d['beam_before_s2']
bp = d['profile']
semigap = d['gap']/2.
beam_offset = d['beam_offset']


for beam_offset in [4.5e-3, 4.6e-3, 4.7e-3]:

    ss = np.linspace(0, (beam0[4,:].max() - beam0[4,:].min())*c*1.01, len_s)
    xx = np.linspace(beam0[0,:].min(), beam0[0,:].max(), len_x)+beam_offset

    distance = semigap - beam_offset

    fig = ms.figure('Test interp around distance %i um' % round(distance*1e6))
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp = sp0 = subplot(sp_ctr, title='Single particle wake', xlabel='s [$\mu$m]', ylabel='Wake [kV/pC/m]')
    sp_ctr += 1

    for index in [0, len_x//2, -1]:
        z = wf_model.wxd(ss, semigap, xx[index])
        distance = semigap - xx[index]
        sp.plot(ss*1e6, z*1e-15, label='%.1f' % (distance*1e6))

    probe_s = beam0[4,:] * c
    probe_x = beam0[0,:] + beam_offset

    wake2d_dict = wf_model.wf2d(probe_s, probe_x, semigap, bp.charge, wf_model.wxd)
    beam_hist = wake2d_dict['beam_hist']
    s_edges = wake2d_dict['s_bins']
    x_edges = wake2d_dict['x_bins']

    sp = sp1 = subplot(sp_ctr, title='Wake potential', xlabel='s [$\mu$m]', ylabel='Wake [MV/m]')
    sp_ctr += 1

    sp = subplot(sp_ctr, title='Current hist', xlabel='s', ylabel='x', scix=True, sciy=True, grid=False)
    sp_ctr += 1

    sp.imshow(np.abs(beam_hist.T), aspect='auto', extent=[s_edges[0], s_edges[-1], x_edges[0], x_edges[-1]], origin='lower')

    wake_from2d = wake2d_dict['wake']

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

plt.show()

