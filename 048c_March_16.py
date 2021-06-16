from socket import gethostname

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-03-16/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/03/16/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-03-16/'



calib_file = data_dir + 
