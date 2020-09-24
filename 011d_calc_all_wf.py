import os
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.io import loadmat
from scipy.constants import c

import data_loader
import uwf_model as uwf
from EmittanceTool.h5_storage import saveH5Recursive

data_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/data_2020-02-03/'

meas1 = 'Bunch_length_meas_2020-02-03_21-54-24.h5'
meas2 = 'Bunch_length_meas_2020-02-03_22-08-38.h5'
meas3 = 'Bunch_length_meas_2020-02-03_23-55-33.h5'

charge_lims1 = [25, 70]
charge_lims2 = [40, 125]
charge_lims3 = [40, 100]

mat_files_current_charge_lims = [
        ('Eloss_UNDbis.mat', meas1, 200e-12, charge_lims1),
        ('Eloss_UND2ndStreak.mat', meas1, 200e-12, charge_lims1),
        ('Eloss_UND-COMP2.mat', meas2, 100e-12, charge_lims2),
        ('Eloss_UND-COMP3.mat', meas3, 100e-12, charge_lims3),
        ]

for mat_file, current_file, charge, lims in mat_files_current_charge_lims:
    bl_meas = data_loader.load_blmeas(data_dir+current_file)
    charge_xx = bl_meas['time_profile1']*c
    charge_xx -= charge_xx.min()
    current_profile = bl_meas['current1']

    charge_profile = current_profile * charge / np.sum(current_profile)
    energy_eV = bl_meas['energy_eV']


    mask_charge = np.logical_and(charge_xx > lims[0]*1e-6, charge_xx < lims[1]*1e-6)
    assert np.any(mask_charge)
    charge_xx = charge_xx[mask_charge]
    charge_xx -= charge_xx.min()
    charge_profile = charge_profile[mask_charge]


    wf_meas = loadmat(data_dir+mat_file)
    gap_list = wf_meas['gap'].squeeze()*1e-3

    def do_calc(gap):
        return uwf.calc_all(charge_xx, charge_profile, gap/2., L=1.)

    with Pool(cpu_count()) as p:
        result_list = p.map(do_calc, gap_list)

    result_dict = {str(i): result_list[i] for i, gap in enumerate(gap_list)}
    result_dict['gap_list'] = gap_list
    result_dict['charge_profile'] = charge_profile
    result_dict['charge_xx'] = charge_xx
    result_dict['energy_eV'] = energy_eV
    save_file = os.path.basename(mat_file) + '_wake.h5'
    saveH5Recursive(save_file, result_dict)
    print('Saved %s' % save_file)



