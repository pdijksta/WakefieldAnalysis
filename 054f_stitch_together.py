from socket import gethostname

import h5_storage

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-05-18/'

data_dir2 = data_dir.replace('18', '19')



files1 = [
        data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', # Affected but ok
        data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5', # Affected but ok
        ]

files2 = [
        data_dir2+'2021_05_19-14_14_22_Calibration_SARUN18-UDCP020.h5',
        data_dir2+'2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5',
        ]


file1, file2 = files1

dict1, dict2 = h5_storage.loadH5Recursive(file1), h5_storage.loadH5Recursive(file2)




