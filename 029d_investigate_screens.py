import pickle
import numpy as np
import matplotlib.pyplot as plt
import myplotstyle as ms

plt.close('all')

with open('./investigate_screen_comp.pkl', 'rb') as f:
    all_screens = pickle.load(f)

reference = all_screens['median_screen']
reference.reshape(2000)
reference.normalize()


ms.figure('Motherfucker')

subplot = ms.subplot_factory(2, 2)
sp_ctr = 1

sp_main = subplot(sp_ctr, title='screens')
sp_ctr += 1


sp_residue = subplot(sp_ctr, title='screens')
sp_ctr += 1



reference.plot_standard(sp_main, lw=3, color='black')

for label, screen in all_screens.items():
    if screen is reference:
        continue
    screen.reshape(2000)
    screen.normalize()

    screen.plot_standard(sp_main, label=label)

    yy_screen = np.interp(reference._xx, screen._xx, screen._yy, left=0, right=0)

    #yy_screen = yy_screen / yy_screen.sum() * reference._yy.sum()

    residue = yy_screen - reference._yy

    sp_residue.plot(residue, label=label)

    print(label, '%e' % reference.compare(screen))

for sp_ in sp_main, sp_residue:
    sp_.legend()


plt.show()
