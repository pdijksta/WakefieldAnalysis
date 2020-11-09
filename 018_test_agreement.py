import pickle
import numpy as np
import matplotlib.pyplot as plt

import myplotstyle as ms

plt.close('all')

with open('./profiles.pkl', 'rb') as f:
    data = pickle.load(f)

s1 = data['meas']
s2 = data['final']

s1.normalize()
s2.normalize()


s1.reshape(1000)
s2.reshape(1000)


c1 = s1.current
c2 = s2.current

t1 = s1.time
t2 = s2.time


def min_func(n_shift0, output='diff'):
    n_shift = int(round(n_shift0))

    if n_shift == 0:
        a1, a2 = c1, c2
    elif n_shift > 0:
        a1, a2 = c1[n_shift:], c2[:-n_shift]
    elif n_shift < 0:
        a1, a2 = c1[:n_shift], c2[-n_shift:]

    diff = np.mean((a1-a2)**2) / np.mean(a1+a2)**2

    if output == 'Full':
        diff, n_shift
    else:
        return diff

step_factor = 0.1
def newton_step(current_x):
    opt_curr = min_func(current_x)
    opt_deriv = min_func(current_x+1) - opt_curr
    delta = - int(round(opt_curr/opt_deriv * step_factor))
    new_x = current_x + delta
    opt_new = min_func(new_x)
    return new_x, opt_curr, opt_new, delta


shift = 0
max_iter = 50
for i in range(max_iter):
    shift, opt_last, opt_new, delta = newton_step(shift)
    print(i, shift, opt_last, opt_new)
    opt = opt_new
    if opt_last < opt_new or abs(delta) < 1:
        print('break')
        shift = shift-delta
        opt = opt_last
        break

n_shift = int(round(shift))
print(n_shift, opt)

if n_shift == 0:
    y1, y2 = c1, c2
    x1, x2 = t1, t2
elif shift > 0:
    y1, y2 = c1[n_shift:], c2[:-n_shift]
    x1, x2 = t1[n_shift:], t2[:-n_shift]
elif shift < 0:
    y1, y2 = c1[:n_shift], c2[-n_shift:]
    x1, x2 = t1[:n_shift], t2[-n_shift:]

x12 = x1 - x1.min() + x2.min()


ms.figure('Compare results')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp = subplot(sp_ctr, title='Profiles')
for x, y, label in [(t1, c1, 'Profile1'), (t2, c2, 'Profile2'), (x12, y1, 'P1 shift')]:
    sp.plot(x, y, label=label)

sp.legend()


plt.show()



