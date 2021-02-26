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
    s, x = np.meshgrid(ss, xx)
    #
    z = wf_model.wxd(s, semigap, x)


    #wf_interp = interp2d(ss, xx, z)

    distance = semigap - beam_offset

    ms.figure('Test interp around distance %i um' % round(distance*1e6))
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp = sp0 = subplot(sp_ctr, title='Single particle wake', xlabel='s [$\mu$m]', ylabel='Wake [kV/pC/m]')
    sp_ctr += 1

    for index in [0, len_x//2, -1]:
        distance = semigap - xx[index]
        sp.plot(ss*1e6, z[index, :]*1e-15, label='%.1f' % (distance*1e6))

    probe_s = beam0[4,:] * c
    probe_x = beam0[0,:] + beam_offset

    beam_hist, s_edges, x_edges = np.histogram2d(probe_s, probe_x, [int(1e3), 100])
    beam_hist *= bp.charge / beam_hist.sum()

    sp = sp1 = subplot(sp_ctr, title='Wake potential', xlabel='s [$\mu$m]', ylabel='Wake [MV/m]')
    sp_ctr += 1

    sp = subplot(sp_ctr, title='Current hist', xlabel='s', ylabel='x', scix=True, sciy=True, grid=False)
    sp_ctr += 1

    sp.imshow(np.abs(beam_hist.T), aspect='auto', extent=[s_edges[0], s_edges[-1], x_edges[0], x_edges[-1]], origin='lower')

    s_edges = s_edges[:-1]
    x_edges = x_edges[:-1] + (x_edges[1] - x_edges[0])*0.5

    wake2d = wf_model.wxd(s_edges[:, np.newaxis], semigap, x_edges)
    output = np.zeros_like(s_edges)

    for n_output in range(len(output)):
        for n2 in range(0, n_output+1):
            output[n_output] += (beam_hist[n2,:] * wake2d[n_output-n2,:]).sum()

    hist1d, _ = np.histogram(probe_s, s_edges)
    hist1d = hist1d / hist1d.sum() * bp.charge
    abs_hist1d = np.abs(hist1d)
    sp.plot(s_edges[:-1], abs_hist1d/abs_hist1d.max()*(x_edges.max()-x_edges.min())*0.3+x_edges.min(), color='red')
    wf_calc = wf_model.WakeFieldCalculator(s_edges, hist1d, bp.energy_eV, 1)
    spw = wf_model.wxd(wf_calc.xx, semigap, beam_offset)
    convoluted_wake = wf_calc.wake_potential(spw)

    sp1.plot(wf_calc.xx*1e6, convoluted_wake/1e6, label='Thin beam')
    sp1.plot(s_edges*1e6, output/1e6, label='Large beam')

    sp0.legend(title='Distance from jaw [$\mu$m]')
    sp1.legend()

plt.show()

