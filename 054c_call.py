import os
import glob
from socket import gethostname

hostname = gethostname()
if hostname == 'desktop':
    dirname1 = '/storage/data_2021-05-18/'
    dirname2 = '/storage/data_2021-05-19/'
elif hostname == 'pc11292.psi.ch':
    dirname1 = '/sf/data/measurements/2021/05/18/'
    dirname2 = '/sf/data/measurements/2021/05/19/'
elif hostname == 'pubuntu':
    dirname1 = '/home/work/data_2021-05-18/'
    dirname2 = '/home/work/data_2021-05-19/'



lasing_files = glob.glob(dirname1+'2021_*Lasing_*SARBD02*.h5')
lasing_files += glob.glob(dirname1+'*Calibration*.h5')

for lasing_file in lasing_files:
    print('%s start' % lasing_file)
    os.system('python3 054b_single.py %s' % lasing_file)
    print('%s done' % lasing_file)
