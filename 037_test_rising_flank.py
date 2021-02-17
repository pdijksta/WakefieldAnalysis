import matplotlib.pyplot as plt
import misc
import pickle

plt.close('all')

with open('./test_rising_flank.pkl', 'rb') as f:
    bp = pickle.load(f)


plt.figure()
sp = plt.subplot(1, 1, 1)

bp.plot_standard(sp, center='Left', label='Left')
bp.plot_standard(sp, center='Right', label='Right')

out1 = misc.find_rising_flank(bp._yy)
out2 = misc.find_rising_flank(bp._yy[::-1])

sp.legend()


plt.show()


