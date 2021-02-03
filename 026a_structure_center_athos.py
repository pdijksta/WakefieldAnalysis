import os
import glob; glob
import matplotlib.pyplot as plt
import numpy as np
#from h5_storage import loadH5Recursive
from scipy.io import loadmat
from scipy.optimize import curve_fit

import myplotstyle as ms


#################
#
#
#  INPUT
#
#
#################


files = ['/sf/data/measurements/2020/10/03/Passive_alignment_20201003T221023.mat',
 '/sf/data/measurements/2020/10/03/Passive_alignment_20201003T214629.mat',
 '/sf/data/measurements/2020/10/03/Passive_alignment_20201003T222227.mat']


bpms_plot = ['SARUN20-DBPM070:X1-RT',] # 'SARBD02-DBPM010:X1-RT']


##################
#
#
# END INPUT
#
#
##################

mean_offset = 0.472
all_files = False
fs_title = 17
fs_label = 15

plt.close('all')

#hostname = socket.gethostname()
#
#if hostname == 'desktop':
#    dirname = '/storage/data_2020-10-03/'
#elif hostname == 'pc11292.psi.ch':
#    dirname = '/sf/data/measurements/2020/10/03/'
#elif hostname == 'pubuntu':
#    dirname = '/home/work/data_2020-10-03/'
#
#
#if all_files:
#    files = sorted(glob.glob(dirname+'Passive_alignment*.mat')[2:])
#else:
#    files = [dirname + f for f in [
#            'Passive_alignment_20201003T221023.mat',
#            'Passive_alignment_20201003T214629.mat',
#            'Passive_alignment_20201003T222227.mat',
#            ]]

fig0 = ms.figure(title='a', figsize=(12, 12))
#fig0.subplots_adjust(wspace=0.4)
subplot = ms.subplot_factory(2,2)
bpm_sp_dict = {}
bpm_sp_dict2 = {}
sp_ctr = 1
for bpm in bpms_plot:
    bpm_sp_dict[bpm] = subplot(sp_ctr, title='', xlabel='Center (mm)', ylabel='Beam position (mm)', grid=False) # , title_fs=fs_title, label_fs=fs_label)
    sp_ctr += 1

fig0 = ms.figure(title='b', figsize=(2, 2))
#fig0.subplots_adjust(wspace=0.4)

for bpm in bpms_plot:
    bpm_sp_dict2[bpm] = subplot(sp_ctr, title='', xlabel='($\mu$m)', ylabel='($\mu$m)', grid=False)
    sp_ctr += 1

#fig_corr = ms.figure('Correlation between SARUN18 and SARUN20 BPMs')
#sp_ctr = 1
#
#
#dict0 = loadmat(dirname + '/Passive_alignment_20201003T213633.mat')
#
#bpmx = [x[0] for x in dict0['BPMX'].squeeze()]
#cen = dict0['center'].squeeze()
#
#index0, index1 = 1, 2
#
#for index in range(4):
#
#    sp_corr = subplot(sp_ctr, title='Center %i mm' % cen[index], xlabel=bpmx[index0]+' ($\mu$m)', ylabel=bpmx[index1]+' ($\mu$m)')
#    sp_ctr += 1
#
#
#    vals1 = dict0['x'][index,:,index0]
#    vals2 = dict0['x'][index,:,index1]
#
#    sp_corr.scatter(vals1*1e3, vals2*1e3)
#



fig1 = ms.figure('Alignment scan')
subplot = ms.subplot_factory(3,4)
sp_ctr = 1

def fit_func(xx, const, strength, wall0, wall1):
    return const + ((xx-wall0)**(-2) - (xx-wall1)**(-2))*strength

offset_list = []
bpm_plot_list = []
bpm_plot_ctr_list = []
gap_list = []

for f in files:


    sp_x = subplot(sp_ctr, title=os.path.basename(f), xlabel='Center (mm)', ylabel='BPM signal')
    sp_ctr += 1
    #sp_y = subplot(sp_ctr, title='Y', xlabel='Center (mm)', ylabel='BPM signal')
    #sp_ctr += 1

    dict_ = loadmat(f)
    print(f, dict_['BPMX'])
    center_arr = dict_['center'].squeeze()
    x_arr = dict_['x']
    y_arr = dict_['y']
    bpms_x = [x[0] for x in dict_['BPMX'].squeeze()]
    bpms_y = [x[0] for x in dict_['BPMY'].squeeze()]

    for dim, bpm_list, arr, sp in [('X', bpms_x, x_arr, sp_x),]: # ('Y', bpms_y, y_arr, sp_y)]:
        for bpm_ctr, bpm in enumerate(bpm_list):
            yy = arr[:,:,bpm_ctr]
            mean = yy.mean(axis=1)
            #n_vals = np.shape(yy)[1]
            #n_uniques = [np.sum(np.unique(yy[i], return_counts=True)[1] != 1) for i in range(len(yy))]
            #print(n_vals, n_uniques)

            p0 = (0, 1, np.min(center_arr)-0.5, np.max(center_arr)+0.5)
            try:
                fit = curve_fit(fit_func, center_arr, mean, p0=p0)
            except RuntimeError:
                raise
                #plt.figure()
                #plt.plot(center_arr, fit_func(center_arr, *p0), label='Guess')
                #plt.plot(center_arr, mean, label='Data')
                #plt.legend()
                #plt.show()
                #import pdb; pdb.set_trace()

            line = sp.errorbar(center_arr, mean, yy.std(axis=1), label=bpm, marker='.', ls='None')
            color = line[0].get_color()

            xx_fit = np.linspace(np.min(center_arr), np.max(center_arr), 100)
            yy_fit = fit_func(xx_fit, *fit[0])
            sp.plot(xx_fit, yy_fit, ls='--', color=color)

            wall0, wall1 = fit[0][2], fit[0][3]
            const = fit[0][0]
            #print(dict_['passive'], center_arr.min(), wall0, wall1, (wall0+wall1)/2)

            gap = abs(wall1-wall0)
            gap_list.append(gap)
            print(gap, int(gap)+1-gap)

            offset = (wall0+wall1)/2
            offset_list.append(offset)
            bpm_plot_list.append(bpm)
            bpm_plot_ctr_list.append(bpm_ctr)

            if bpm in bpms_plot and ('227.mat' in f or '023.mat' in f or '629.mat' in f):
                for sp_, factor in [(bpm_sp_dict[bpm], 1), (bpm_sp_dict2[bpm], 1e3)]:
                    gap_label = '%.2f' % gap_list[-1]
                    color = sp_.errorbar((center_arr-mean_offset)*factor, (mean-const)*factor, yy.std(axis=1), marker='.', ls='None', label=gap_label)[0].get_color()
                    sp_.plot((xx_fit-mean_offset)*factor, (yy_fit-const)*factor, ls='--', color=color)
                    print(offset-mean_offset, factor)
                    if factor == 1e3:
                        sp_.scatter((offset-mean_offset)*factor, 0, color=ms.colorprog(bpm_ctr, bpm_list), marker='x')



    sp_x.legend()

for sp in bpm_sp_dict.values():
    sp.legend(title='Gap (mm)', fontsize=fs_label)
    #sp.set_xlim(-0.5, 0.5)
    sp.tick_params(axis='x', labelsize=fs_label)
    sp.tick_params(axis='y', labelsize=fs_label)

for sp in bpm_sp_dict2.values():
    sp.set_xlim(-5, 5)
    sp.set_ylim(-0.2, 0.2)

offset_arr = np.array(offset_list)*1e3

ms.figure('Summary')
subplot = ms.subplot_factory(1,1)
sp = subplot(1, title='Summary', xlabel='Gap', ylabel='Structure offset fit [$\mu$m]')
labelled_ctrs = set()
for gap, offset, bpm, bpm_ctr in zip(gap_list, offset_arr, bpm_plot_list, bpm_plot_ctr_list):
    if bpm_ctr not in labelled_ctrs:
        label = bpm
        labelled_ctrs.add(bpm_ctr)
    else:
        label = None
    sp.scatter(gap, offset, color=ms.colorprog(bpm_ctr, bpm_list), label=label)

sp.axhline(offset_arr.mean(), color='black', ls='--')
sp.legend()

ms.saveall('/tmp/offset_for_paper', ending='.pdf')


plt.show()


