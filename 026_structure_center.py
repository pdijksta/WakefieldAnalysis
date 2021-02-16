import os
import glob; glob
import socket
import matplotlib.pyplot as plt
import numpy as np
#from h5_storage import loadH5Recursive
from scipy.io import loadmat
from mat73 import loadmat as loadmat73
from scipy.optimize import curve_fit

import myplotstyle as ms


mean_offset = 0.472
fs_title = 17
fs_label = 15

plt.close('all')

hostname = socket.gethostname()

if hostname == 'desktop':
    dirname = '/storage/data_2020-10-03/'
elif hostname == 'pc11292.psi.ch':
    dirname = '/sf/data/measurements/2020/10/03/'
elif hostname == 'pubuntu':
    dirname = '/home/work/data_2020-10-03/'

dirname2 = dirname.replace('03', '04')

files = [dirname + f for f in [
        'Passive_alignment_20201003T221023.mat',
        'Passive_alignment_20201003T214629.mat',
        'Passive_alignment_20201003T222227.mat',
        ]] + [dirname2 + f for f in [
            #'Passive_data_20201004T172425.mat',
            ]]

fig0 = ms.figure(title='a', figsize=(12, 12))
#fig0.subplots_adjust(wspace=0.4)
subplot = ms.subplot_factory(2,2)
bpms_plot = ['SARBD01-DBPM040:X1-RT', 'SARBD02-DBPM010:X1-RT', 'SARBD02-DBPM040:X1-RT']
bpm_sp_dict = {}
bpm_sp_dict2 = {}
sp_ctr = 1
for bpm in bpms_plot:
    bpm_sp_dict[bpm] = subplot(sp_ctr, title=bpm, xlabel='Center (mm)', ylabel='Beam position (mm)', grid=False) # , title_fs=fs_title, label_fs=fs_label)
    sp_ctr += 1

fig0 = ms.figure(title='b', figsize=(2, 2))
sp_ctr = 1
#fig0.subplots_adjust(wspace=0.4)

for bpm in bpms_plot:
    bpm_sp_dict2[bpm] = subplot(sp_ctr, title=bpm, xlabel='($\mu$m)', ylabel='($\mu$m)', grid=False)
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
subplot = ms.subplot_factory(2, 2)
sp_ctr = 1

order = order0 = 3

def fit_func(xx, const, strength, wall0, wall1):
    if order == 2:
        return const + ((xx-wall0)**(-2) - (xx-wall1)**(-2))*strength
    elif order == 3:
        return const + ((xx-wall0)**(-3) + (xx-wall1)**(-3))*strength


offset_list = []
bpm_plot_list = []
bpm_plot_ctr_list = []
gap_list = []

for f in files:
    #if dirname2 not in f:
    #    continue


    sp_x = subplot(sp_ctr, title=os.path.basename(f), xlabel='Center (mm)', ylabel='BPM signal')
    sp_ctr += 1
    #sp_y = subplot(sp_ctr, title='Y', xlabel='Center (mm)', ylabel='BPM signal')
    #sp_ctr += 1

    try:
        dict_ = loadmat(f)
    except NotImplementedError:
        dict_ = loadmat73(f)
        dict2 = {}
        for x, y in dict_.items():
            try:
                dict2[x] = np.array(y)
            except:
                dict2[x] = y
        dict_ = dict2
        dict_['BPMX'] = dict_['BPMX'].tolist()
        dict_['BPMY'] = dict_['BPMY'].tolist()

    #print(f, dict_['BPMX'])
    if 'center' in dict_:
        center_arr = dict_['center'].squeeze()
        bpms_x = [x[0] for x in dict_['BPMX'].squeeze()]
        bpms_y = [x[0] for x in dict_['BPMY'].squeeze()]
    else:
        center_arr = dict_['value'].squeeze()
        bpms_x = dict_['BPMX']
        bpms_y = dict_['BPMY']
    x_arr = dict_['x']
    y_arr = dict_['y']

    for dim, bpm_list, arr, sp in [('X', bpms_x, x_arr, sp_x),]: # ('Y', bpms_y, y_arr, sp_y)]:
        for bpm_ctr, bpm in enumerate(bpm_list):
            #print(f, bpm)
            yy = arr[:,:,bpm_ctr]
            mean = yy.mean(axis=1)
            #n_vals = np.shape(yy)[1]
            #n_uniques = [np.sum(np.unique(yy[i], return_counts=True)[1] != 1) for i in range(len(yy))]
            #print(n_vals, n_uniques)

            p0 = (0, 1, np.min(center_arr)-0.5, np.max(center_arr)+0.5)
            try:
                order = 2
                fit2 = curve_fit(fit_func, center_arr, mean, p0=p0)
                chisq2 = np.sum((mean - fit_func(center_arr, *fit2[0]))**2)
                order = 3
                fit3 = curve_fit(fit_func, center_arr, mean, p0=p0)
                chisq3 = np.sum((mean - fit_func(center_arr, *fit3[0]))**2)

                if order0 == 2:
                    fit = fit2
                elif order0 == 3:
                    fit = fit3

                print(os.path.basename(f), bpm, chisq2, chisq3)

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

            xx_fit = np.linspace(np.min(center_arr), np.max(center_arr), 1000)
            yy_fit = fit_func(xx_fit, *fit[0])
            sp.plot(xx_fit, yy_fit, ls='--', color=color)

            wall0, wall1 = fit[0][2], fit[0][3]
            const = fit[0][0]
            #print(dict_['passive'], center_arr.min(), wall0, wall1, (wall0+wall1)/2)

            gap = abs(wall1-wall0)
            if gap > 30:
                gap = np.nan
            gap_list.append(gap)
            #print(gap, int(gap)+1-gap)

            offset = (wall0+wall1)/2
            #print('%s %s %.5e' % (f, bpm, offset))
            offset_list.append(offset)
            bpm_plot_list.append(bpm)
            bpm_plot_ctr_list.append(bpm_ctr)

            if bpm in bpms_plot:
                print(f, bpm, 'a')
                for sp_, factor in [(bpm_sp_dict[bpm], 1), (bpm_sp_dict2[bpm], 1e3)]:
                    gap_label = '%.2f' % gap_list[-1]
                    color = sp_.errorbar((center_arr-mean_offset)*factor, (mean-const)*factor, yy.std(axis=1), marker='.', ls='None', label=gap_label)[0].get_color()
                    sp_.plot((xx_fit-mean_offset)*factor, (yy_fit-const)*factor, ls='--', color=color)
                    #print(offset-mean_offset, factor)
                    if factor == 1e3:
                        sp_.scatter((offset-mean_offset)*factor, 0, color=color, marker='x')



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


