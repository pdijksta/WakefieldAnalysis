import numpy as np
import pickle

import wf_model
import myplotstyle as ms

ms.plt.close('all')

with open('./test_wake_conv.pkl', 'rb') as f:
    d = pickle.load(f)

with open('./s_edges.pkl', 'rb') as f:
    s_edges = pickle.load(f)

beam_hist = d['beam_hist']
x_edges = d['x_edges']
wake = np.zeros_like(beam_hist)

wake2d = wf_model.wxq(s_edges[:, np.newaxis], 5e-3, x_edges)

#for n_output in range(10):
#    for nx in range(wake.shape[1]):
#        for n2 in range(0, n_output+1):
#            for nxp in range(wake.shape[1]):
#                wake[n_output,nx] += beam_hist[n2,nxp] * wake2d[n_output-n2,nxp] * (x_edges[nx] - x_edges[nxp])
#    print('%i done' % n_output)


wake_opt = np.zeros_like(wake)

for n_output in range(wake_opt.shape[0]):
    for n2 in range(0, n_output+1):
        wake0 = beam_hist[n2,:] * wake2d[n_output-n2,:]
        c1 = wake0.sum() * x_edges
        c2 = (wake0 * x_edges).sum()
        wake_opt[n_output,:] += c1 - c2
    #print('%i done' % n_output)


wake = wake_opt

sp_wake_old = wf_model.wxq(s_edges, 5e-3, x_edges.mean())
current = np.sum(beam_hist, axis=1)

wake_convolve = np.convolve(current, sp_wake_old)[:len(current)]
wake_old = wake_convolve[:, np.newaxis] * (x_edges - x_edges.mean())


fig = ms.figure('Beam size effect on quad wake')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp = subplot(sp_ctr, grid=False, title='Old', scix=True, sciy=True)
sp_ctr += 1

out = sp.imshow(np.abs(wake_old*1e-3).T, aspect='auto', extent=[s_edges[0], s_edges[-1], x_edges[0], x_edges[-1]], origin='lower')
fig.colorbar(out, ax=sp, extend='both')

sp = subplot(sp_ctr, grid=False, title='New', scix=True, sciy=True)
sp_ctr += 1

out = sp.imshow(np.abs(wake*1e-3).T, aspect='auto', extent=[s_edges[0], s_edges[-1], x_edges[0], x_edges[-1]], origin='lower')
fig.colorbar(out, ax=sp, extend='both')


sp = subplot(sp_ctr, title='Sum of abs(wake)', scix=True, sciy=True)
sp_ctr += 1

sp.plot(s_edges, np.abs(wake).mean(axis=1), label='New')
sp.plot(s_edges, np.abs(wake_old).mean(axis=1), label='Old')

sp.legend()




ms.plt.show()




