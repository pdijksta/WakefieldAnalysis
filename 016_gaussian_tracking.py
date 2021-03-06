import numpy as np
import matplotlib.pyplot as plt
import elegant_matrix
#import wf_model
import myplotstyle as ms
import h5_storage

plt.close('all')

fig = ms.figure('Gaussian generation')
sp_ctr = 1
subplot = ms.subplot_factory(3,3)

sp_current = subplot(sp_ctr, title='Current')
sp_ctr += 1

full_gap = 6e-3
offset0 = -2.63e-3
hist_bin_size = 8.7e-6
del_offset_arr = np.array([-40, -20, 0, 20, 40])*2*1e-6
n_particles = int(300e3)
energy_eV = 5.95e9
use_constant_bin_number = True

if use_constant_bin_number:
    bin_number = 200
else:
    bin_number = 6500

sigma_arr = np.array([6.5, 13.3, 40])*1e-15
arr0 = np.zeros([len(del_offset_arr), len(sigma_arr), bin_number], float)
output = {
        'energy_eV': energy_eV,
        'full_gap_s2': full_gap,
        'delta_offset': del_offset_arr,
        'proj_no_streak': arr0.copy(),
        'axis_no_streak': arr0.copy(),
        'proj_streak': arr0.copy(),
        'axis_streak': arr0.copy(),
        }
timestamp = elegant_matrix.get_timestamp(2020, 2, 3, 21, 54, 24)
xx = np.linspace(0., 200e-15, 1000)

for n_sigma, sigma in enumerate(sigma_arr):
    curr = np.exp(-(xx-np.mean(xx))**2/(2*sigma**2))
    curr[curr<0.001*curr.max()] = 0

    sp_current.plot(xx, curr/curr.max())
    simulator = elegant_matrix.get_simulator('/afs/psi.ch/intranet/SF/Beamdynamics/Philipp//data/archiver_api_data/2020-02-03.json11')

    for n_offset, del_offset in enumerate(del_offset_arr):

        if n_offset == 0:
            sim0, mat0, _ = simulator.simulate_streaker(xx, curr, timestamp, (20e-3, full_gap), (0, 0), energy_eV, n_particles=n_particles, linearize_twf=False)
            #print(sigma*1e15, del_offset*1e6, sim0.watch[-1]['x'].std())
        sim1, mat1, _ = simulator.simulate_streaker(xx, curr, timestamp, (20e-3, full_gap), (0, offset0+del_offset), energy_eV, n_particles=n_particles, linearize_twf=False)

        if n_offset == 0 and n_sigma == 0:
            r0 = mat0['MIDDLE_STREAKER_2']
            r1 = mat0['SARBD02.DSCR050']

            rr = np.matmul(r1, np.linalg.inv(r0))
            output['R12_s2_to_screen'] = rr[0,1]

        for sim, label in [(sim0, 'no_streak'), (sim1, 'streak')]:
            w = [x for x in sim.watch if 'sarbd02' in x.filename][0]
            print(w.filename)

            if sp_ctr > 9:
                ms.figure('Continued')
                sp_ctr = 1

            sp = subplot(sp_ctr, title='%i %s' % (sigma*1e15, label), scix=True)
            sp_ctr += 1

            if use_constant_bin_number:
                bin_edges = bin_number
            else:
                bin_edges = np.arange(w['x'].min(), w['x'].max()+hist_bin_size, hist_bin_size)

            hist, bin_edges, _ = sp.hist(w['x'], density=True, bins=bin_edges)

            output['proj_'+label][n_offset, n_sigma, :len(bin_edges)-1] = hist
            output['axis_'+label][n_offset, n_sigma, :len(bin_edges)-1] = bin_edges[:-1]

#sim0.__del__()
#sim.__del__()

h5_storage.saveH5Recursive('./bins_200_300e3_370um_larger_beamsize.h5', output)
plt.show()

