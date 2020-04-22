import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import data_loader

import myplotstyle as ms

plt.close('all')

dl = data_loader.DataLoader(file_json='/storage/data_2020-02-03/2020-02-03.json1')
var = 'SARBD01-MBND100:P-SET'

energy = dl[var][:,1]
time = dl[var][:,0]

fig = ms.figure('Energy during shift')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp = subplot(sp_ctr, title='Energy', ylabel=var)
time_obj = [datetime.fromtimestamp(i) for i in time]

sp.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%H'))
sp.plot(time_obj, energy)

plt.show()

