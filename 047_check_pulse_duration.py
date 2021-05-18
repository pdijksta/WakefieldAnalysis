import h5_storage
import image_and_profile as iap

file_ = '/home/work/tmp_reconstruction/2021_04_26-12_23_40_PassiveReconstruction.h5'

dd = h5_storage.loadH5Recursive(file_)
rec_profile = iap.BeamProfile.from_dict(dd['gaussian_reconstruction']['reconstructed_profile'])

