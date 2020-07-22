import matplotlib.pyplot as plt
import datetime

import elegant_matrix
from ElegantWrapper.simulation import ElegantSimulation

import myplotstyle as ms

plt.close('all')


year, month, day, hour, minute, second = 2020, 2, 9, 22, 14, 54
date = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
timestamp = int(date.strftime('%s'))
mat0 = elegant_matrix.get_elegant_matrix(0, timestamp, del_sim=False, print_=False)
mat1 = elegant_matrix.get_elegant_matrix(1, timestamp, del_sim=False, print_=False)


dscr0 = 'SARBD01.DSCR050'
dscr1 = 'SARBD02.DSCR050'
print(mat0[dscr0])
print(mat1[dscr0])


print(mat0[dscr1])
print(mat1[dscr1])


sim0 = ElegantSimulation('./for_alex//elegant_5521_2020-06-19_17-17-32_0/SwissFEL_in0.ele')
sim1 = ElegantSimulation('./for_alex//elegant_5521_2020-06-19_17-17-33_1/SwissFEL_in0.ele')


ms.figure()
subplot = ms.subplot_factory(2,2)

sp_ctr = 1

for dim in 'x', 'y':
    sp = subplot(sp_ctr, title='Beta_%s' % dim)
    sp_ctr += 1
    for n_sim, sim in enumerate([sim0, sim1]):
        twi = sim.twi
        sp.plot(twi['s'], twi['beta%s' % dim], label=n_sim)
    sp.set_xlim(None, 25)
    sp.legend()

sp = subplot(sp_ctr, title='R12')
sp_ctr += 1
for n_sim, sim in enumerate([sim0, sim1]):
    mat = sim.mat
    ss = mat['s']
    sp.plot(ss, mat['R12'], label=n_sim)

for dscr in (dscr0, dscr1):
    s_dscr = sim.get_element_position(dscr, mean=True)
    sp.axvline(s_dscr, label=dscr)
sp.set_xlim(None, 25)
sp.legend()

plt.show()



