import matplotlib.pyplot as plt
import analysis
from h5_storage import loadH5Recursive

import myplotstyle as ms

plt.close('all')

filename = './2021_03_16-18_50_47_Calibration_data_SARUN18-UDCP020.h5'
dict_ = loadH5Recursive(filename)

output = analysis.analyze_streaker_calibration(filename)
print(output['meta_data']['streaker_offset']*1e6, 'um')


ms.figure('Images')

subplot = ms.subplot_factory(2,2, grid=False)
sp_ctr = 1

for n_img in range(4):
    sp_img = subplot(sp_ctr)
    sp_ctr += 1

    image0 = dict_['pyscan_result']['image'][n_img].astype(float)

    sp_img.imshow(image0, aspect='auto')







plt.show()

