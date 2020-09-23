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
offset = 2.5e-3


sigma_arr = np.array([10, 5, 25])*1e-15
output = {
        'full_gap_s2': full_gap,
        'beam_offset_s2': offset,
        'proj_no_streak': [],
        'axis_no_streak': [],
        'proj_streak': [],
        'axis_streak': [],
        'R12_s2_to_screen': [],
        }
for n_sigma, sigma in enumerate(sigma_arr):
    xx = np.linspace(0., 200e-15, 1000)
    curr = np.exp(-(xx-np.mean(xx))**2/(2*sigma**2))

    sp_current.plot(xx, curr/curr.max())


    simulator = elegant_matrix.get_simulator('/afs/psi.ch/intranet/SF/Beamdynamics/Philipp//data/archiver_api_data/2020-02-03.json11')

    timestamp = elegant_matrix.get_timestamp(2020, 2, 3, 21, 54, 24)
    sim0, mat0 = simulator.simulate_streaker(xx, curr, timestamp, (20e-3, full_gap), (0, 0))
    sim1, mat1 = simulator.simulate_streaker(xx, curr, timestamp, (20e-3, full_gap), (0, offset))

    r0 = mat0['MIDDLE_STREAKER_2']
    r1 = mat0['SARBD02.DSCR050']

    rr = np.matmul(r1, np.linalg.inv(r0))
    output['R12_s2_to_screen'].append(rr[0,1])

    for sim, label in [(sim0, 'no_streak'), (sim1, 'streak')]:
        w = sim.watch[-1]
        print(w.filename)

        sp = subplot(sp_ctr, title='%i %s' % (sigma*1e15, label), scix=True)
        sp_ctr += 1

        hist, bin_edges, _ = sp.hist(w['x'], density=True, bins=200)

        output['proj_'+label].append(hist)
        output['axis_'+label].append(bin_edges[:-1])


h5_storage.saveH5Recursive('./three_cases.h5', output)



plt.show()

