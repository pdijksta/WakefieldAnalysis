import os
from scipy.constants import c
import numpy as np
import matplotlib.pyplot as plt

from ElegantWrapper.simulation import ElegantSimulation
from ElegantWrapper.watcher import Watcher2
import data_loader
import wf_model

import myplotstyle as ms


plt.close('all')



n_particles = int(20e3) # for elegant
ele_dir0 = '/home/philipp/elegant/wakefield/001_cut_sim/'
ele_dir = '/home/philipp/elegant/wakefield/001a_test_measured_current_prof/'

# Shift day was 3.2.2020


bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_21-54-24.h5'

bl_meas = data_loader.load_blmeas(bl_meas_file)

tt = bl_meas['time_profile1']
tt -= tt.mean()

curr0 = bl_meas['current1']
curr = curr0.copy()

curr[curr<curr.max()*0.05] = 0

integrated_curr = np.cumsum(curr)
integrated_curr /= integrated_curr.max()


randoms = np.random.rand(n_particles)
interp_tt = np.interp(randoms, integrated_curr, tt)





ms.figure('Generate sdds beam')
plt.subplots_adjust(hspace=0.3, wspace=0.3)
subplot = ms.subplot_factory(3,3)
sp_ctr = 1

sp_hist = subplot(sp_ctr, title='Histogram of generated')
sp_ctr += 1

hist, bin_edges = np.histogram(interp_tt, bins=200, normed=True)

sp_hist.hist(interp_tt, bins=200, density=True, label='interp_tt')

factor = np.trapz(curr0, tt)

sp_hist.plot(tt, curr0/factor, label='curr0')


sim0 = ElegantSimulation(ele_dir0 + 'SwissFEL_in1.ele')
w = sim0.watch[0]


sp_hist.hist(w['t'] - w['t'].mean(), bins=200, density=True, label='Orig')

sp_hist.legend()

new_watcher_dict = {}
for key in ('t', 'p', 'x', 'y', 'xp', 'yp'):
    new_watcher_dict[key] = w[key]

new_watcher_dict['t'] = interp_tt
#new_watcher_dict['p'] = np.ones_like(w['p']) * w['p'].mean()
new_watcher_dict['xp'] -= new_watcher_dict['xp'].mean()

new_watcher = Watcher2({}, new_watcher_dict)

new_watcher.toSDDS(ele_dir + 'test_new_sdds.sdds')


max_xx = (interp_tt.max() - interp_tt.min())*c*1.1

xx = np.linspace(0, max_xx, int(10000))

displacement = (3.0884-0.65)*1e-3

generated_wf = wf_model.generate_elegant_wf(ele_dir + '/passive_streaker_wake.sdds', xx, 6e-3/2., displacement, L=1)
beamsize = 15e-6

generated_wf_minus = wf_model.generate_elegant_wf(ele_dir + '/passive_streaker_wake.sdds', xx, 6e-3/2., displacement-beamsize, L=1)
generated_wf_plus = wf_model.generate_elegant_wf(ele_dir + '/passive_streaker_wake.sdds', xx, 6e-3/2., displacement+beamsize, L=1)


sp = subplot(sp_ctr, title='Generated wakes')
sp_ctr += 1
sp.plot(generated_wf['t'], generated_wf['W'], label='W')
sp.plot(generated_wf['t'], generated_wf['WX'], label='WX')
sp.plot(generated_wf['t'], generated_wf_minus['WX'], label='WX-')
sp.plot(generated_wf['t'], generated_wf_plus['WX'], label='WX+')
sp.legend()


old_dir = os.getcwd()
try:
    os.chdir(ele_dir)
    os.system('elegant SwissFEL_in1.ele')
finally:
    os.chdir(old_dir)

sim = ElegantSimulation(ele_dir + 'SwissFEL_in1.ele')

sp = subplot(sp_ctr, title='Image at SARBD01-DSCR050', grid=False, scix=True, sciy=True, xlabel='x [m]', ylabel='y [m]')
sp_ctr += 1

w = sim.watch[-2]

sp.hist2d(w['x'], w['y'], bins=100)




sp = subplot(sp_ctr, title='Image at SARBD02-DSCR050', grid=False, scix=True, sciy=True, xlabel='x [m]', ylabel='y [m]')
sp_ctr += 1

w = sim.watch[-1]

sp.hist2d(w['x'], w['y'], bins=100)


sp = subplot(sp_ctr, title='Phase space at SARBD02-DSCR050', grid=False, scix=True, sciy=True, xlabel='z [m]', ylabel='Energy [MeV')
sp_ctr += 1

sp.hist2d((w['t']-w['t'].mean())*c, (w['p']-w['p'].mean())*511e3*1e-6, bins=100)


sp = subplot(sp_ctr, title='Projected', scix=True)
sp_ctr += 1
sp.hist(w['x'], bins=100)


sp = subplot(sp_ctr, title='Centroid', sciy=True)
sp_ctr +=1
sp.plot(sim.cen['s'], sim.cen['Cx'], label='x')
sp.plot(sim.cen['s'], sim.cen['Cy'], label='y')

udcp_pos = sim.get_element_position('SARUN18.UDCP020', mean=True)
sp.axvline(udcp_pos, color='black', label='Second streaker')
sp.legend()

sp = subplot(sp_ctr, title='Mean energy loss', sciy=True, ylabel='Energy loss [MeV]')
sp_ctr += 1
sp.plot(sim.cen['s'], (sim.cen['pCentral']-sim.cen['pCentral'][0])*511e3/1e6)

sp = subplot(sp_ctr, title='Current first and last watcher', xlabel='z [$\mu$m]')
sp_ctr += 1

for w_index, label in [(0, 'first'), (2, 'middle'), (-1, 'last')]:
    w = sim.watch[w_index]
    sp.hist((w['t']-w['t'].mean())*c*1e6, bins=200, label=label+' at %i m' % w.s, density=True)

sp.legend()



plt.show()

