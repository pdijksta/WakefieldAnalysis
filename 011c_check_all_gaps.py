from scipy.io import loadmat
data_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/data_2020-02-03/'

mat_files = [
        'Eloss_UNDbis.mat',
        'Eloss_UND2ndStreak.mat',
        'Eloss_DEH1-COMP1.mat',
        'Eloss_DEH1-COMP2.mat',
        'Eloss_DEH1-COMP2.mat',
        'Eloss_UND-COMP3.mat',
        'Eloss_DEH1-COMP3.mat',
        ]


for mat in mat_files:
    file_ = data_dir+mat
    dd = loadmat(file_)
    print(mat, dd['gap'])

