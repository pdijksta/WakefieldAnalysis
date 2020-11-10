from copy import deepcopy
import pickle
#import numpy as np
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


s3 = deepcopy(s1)
result = s3.find_agreement2(s2, 1e-13)

ms.figure('Compare results')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp = subplot(sp_ctr, title='Profiles')
for profile, label in [(s1, 'Profile1'), (s2, 'Profile2'), (s3, 'P1 shift')]:
    sp.plot(profile._xx, profile._yy, label=label)

sp.legend()


plt.show()

