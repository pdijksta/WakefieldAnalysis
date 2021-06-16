import matplotlib.pyplot as plt
import h5_storage
import analysis

plt.close('all')

file_on = '/sf/data/measurements/2021/05/18/2021_05_18-18_13_33_Lasing_True_SARBD02-DSCR050.h5'
file_off = '/sf/data/measurements/2021/05/18/2021_05_18-18_27_23_Lasing_False_SARBD02-DSCR050.h5'

dict_on = h5_storage.loadH5Recursive(file_on)

energy_eV = dict_on['meta_data_end']['SARBD01-MBND100:ENERGY-OP']*1e6

screen_center = 900e-6
structure_center = 370e-6

structure_length = 1

file_current = '/sf/data/measurements/2021/05/18/2021_05_18-19_26_41_PassiveReconstruction.h5'

r12 = 7.130170460315602
disp = 0.439423807827296
streaker = 'SARUN18-UDCP020'
pulse_energy = 600e-6
charge = 200e-12

obj = analysis.Reconstruction(screen_center, [0, structure_center])

analysis.reconstruct_lasing(file_on, file_off, screen_center, structure_center, structure_length, file_current, r12, disp, energy_eV, charge, streaker, None, pulse_energy)


plt.show()

