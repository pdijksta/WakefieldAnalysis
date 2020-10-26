import numpy as np
import matplotlib.pyplot as plt
import elegant_matrix
import data_loader
import myplotstyle as ms
import h5_storage

plt.close('all')

fig = ms.figure('Track current beam')
sp_ctr = 1
subplot = ms.subplot_factory(3,3)

sp_current = subplot(sp_ctr, title='Current')
sp_ctr += 1

full_gap = 10e-3
offset0 = (-4.22e-3) - 0.52e-3
del_offset_arr = np.array([0,])*2*1e-6
n_particles = int(20e3)
energy_eV = 5.95e9
bin_number = 200

timestamp = elegant_matrix.get_timestamp(2020, 10, 3, 15, 43, 29)
timestamp = 1601681166

dl = data_loader.load_blmeas('/sf/data/measurements/2020/10/03/Bunch_length_meas_2020-10-03_15-43-29.h5')
energy_eV = dl["energy_eV"]

arr0 = np.zeros([len(del_offset_arr), bin_number], float)
output = {
        'energy_eV': energy_eV,
        'full_gap_s2': full_gap,
        'delta_offset': del_offset_arr,
        'proj_no_streak': arr0.copy(),
        'axis_no_streak': arr0.copy(),
        'proj_streak': arr0.copy(),
        'axis_streak': arr0.copy(),
        }

curr = dl['current1'][::-1]
xx = dl['time_profile1']

sp_current.plot(xx, curr/curr.max())
simulator = elegant_matrix.get_simulator('/afs/psi.ch/intranet/SF/Beamdynamics/Philipp//data/archiver_api_data/2020-10-03.json1')

for n_offset, del_offset in enumerate(del_offset_arr):

    if n_offset == 0:
        sim0, mat0 = simulator.simulate_streaker(xx, curr, timestamp, (20e-3, full_gap), (0, 0), energy_eV, n_particles=n_particles, linearize_twf=False)[:2]
        #print(sigma*1e15, del_offset*1e6, sim0.watch[-1]['x'].std())
    sim1, mat1 = simulator.simulate_streaker(xx, curr, timestamp, (20e-3, full_gap), (0, offset0+del_offset), energy_eV, n_particles=n_particles, linearize_twf=False)[:2]

    if n_offset == 0:
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

        sp = subplot(sp_ctr, title=label, scix=True)
        sp_ctr += 1

        bin_edges = bin_number

        hist, bin_edges, _ = sp.hist(w['x'], density=True, bins=bin_edges)

        output['proj_'+label][n_offset,:len(bin_edges)-1] = hist
        output['axis_'+label][n_offset,:len(bin_edges)-1] = bin_edges[:-1]

#sim0.__del__()
#sim.__del__()

#h5_storage.saveH5Recursive('./bins_200_300e3_370um_larger_beamsize.h5', output)
plt.show()

