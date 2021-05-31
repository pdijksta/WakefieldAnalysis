import matplotlib.pyplot as plt
from socket import gethostname

from h5_storage import loadH5Recursive
import analysis

plt.close('all')

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-05-18/'


#calib_data = loadH5Recursive(data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5')['raw_data']
#calib_data2 = loadH5Recursive(data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5')['raw_data']
#
#calib_dict = analysis.analyze_streaker_calibration(calib_data, do_plot=True, fit_order=False)['meta_data']
#calib_dict2 = analysis.analyze_streaker_calibration(calib_data2, do_plot=True, fit_order=False)['meta_data']
#
#print('%i' % (calib_dict['streaker_offset']*1e6), calib_dict['order_fit'], True)
#print('%i' % (calib_dict2['streaker_offset']*1e6), calib_dict2['order_fit'], False)


calib_data = loadH5Recursive(data_dir+'2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5')['raw_data']
calib_dict = analysis.analyze_streaker_calibration(calib_data, do_plot=True, fit_order=False)['meta_data']



plt.show()

