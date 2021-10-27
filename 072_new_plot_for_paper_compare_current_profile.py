from socket import gethostname

import blmeas
import streaker_calibration
import image_and_profile as iap
import myplotstyle as ms

ms.closeall()

charge = 180e-12
energy_eV = 6e9

hostname = gethostname()
if hostname == 'desktop':
    data_dir1 = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir1 = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir1 = '/mnt/data/data_2021-05-18/'

data_dir2 = data_dir1.replace('05', '10')
data_dir3 = data_dir2.replace('18', '24')

blmeas_file1 = data_dir1+'119325494_bunch_length_meas.h5'
blmeas_file2 = data_dir2+'132380133_bunch_length_meas.h5'
blmeas_file3 = data_dir3+'132880221_bunch_length_meas.h5'

sc_file1 = data_dir1+'2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5'
sc_file2 = data_dir2+'2021_10_18-11_13_17_Calibration_data_SARUN18-UDCP020.h5'
sc_file3 = data_dir3+'2021_10_24-10_34_00_Calibration_SARUN18-UDCP020.h5'


blmeas_files = [blmeas_file1, blmeas_file2, blmeas_file3]
sc_files = [sc_file1, sc_file2, sc_file3]
days = ['May 18', 'October 18', 'October 24']

ms.figure('Bunch length meas')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_current = sp = subplot(sp_ctr, title='Bunch length measurements')
sp_ctr += 1

blmeas_dicts = []


for ctr, (blmeas_file, day) in enumerate(zip(blmeas_files, days)):
    #if ctr == 0:
    #    profile = iap.profile_from_blmeas(blmeas_file, 200e-15, charge, energy_eV, True)
    #else:
    blmeas_dict = blmeas.load_avg_blmeas(blmeas_file)
    blmeas_dicts.append(blmeas_dict)
    profile = iap.BeamProfile(blmeas_dict[1]['time'][::-1], blmeas_dict[2]['current_reduced'][::-1], energy_eV, charge)
    profile.plot_standard(sp, label=day)

sp.legend()

for sc_file, day in zip(sc_files, days):
    sc = streaker_calibration.StreakerCalibration('Aramis', 1, 10e-3, charge)
    #sc.add_



ms.show()

