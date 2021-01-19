#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:31:40 2020

@author: alex
"""
import h5py
import os
import matplotlib.pyplot as plt
import math as m
from random import seed
from random import random
# seed random number generator
seed(1)

import scipy.integrate as integrate
import scipy.special as special


import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp


import numpy as np


#filenameTDC = 'Bunch_length_meas_2020-07-26_15-03-59.h5' #July26

#filenameTDC ='129858802_bunch_length_meas.h5' #October 4 night

filenameTDC ='/storage/data_2020-10-03/'+'129833611_bunch_length_meas.h5' #October 4 night


def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
def distS(s,sigma,s0):
    return m.exp(-((s-s0)**2/(2*sigma**2)))/m.sqrt(2*m.pi)/sigma

def sigmafit(x,y):
    mean = sum(x*y)/np.sum(y)
    sigma = m.sqrt(abs(sum(y*(x-mean)**2)/np.sum(y)))  
    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma],maxfev=2000)
    return popt

def readTDC(filename):

    fTDC = h5py.File(filename, 'r')
    #print("Keys: %s" % fTDC.keys())
    #print("SubKeys: %s" % fTDC['Raw_data'].keys())
    #print("SubKeys: %s" % fTDC['Meta_data'].keys())
    #PSTDCall = np.array(fTDC['Raw_data']['Beam images'])
    Ic1all = np.array(fTDC['Meta_data']['Current profile'])
    Ic1all2 = np.array(fTDC['Meta_data']['Current profile 2'])
    Ifit1 = np.array(fTDC['Meta_data']['Fit current profile'])
    time1 = np.array(fTDC['Meta_data']['Time axes'])
    time2 = np.array(fTDC['Meta_data']['Time axes 2'])
    cf1= np.array(fTDC['Meta_data']['Calibration factor 1'])
    cf2= np.array(fTDC['Meta_data']['Calibration factor 2'])
    #print(np.shape(Ic1all))
    I1=Ic1all-np.min(Ic1all)
    I2=Ic1all2-np.min(Ic1all2)
    t01=sum(time1*I1)/np.sum(I1)
    t02=sum(time2*I2)/np.sum(I2)
    
    time1=time1-t01
    time2=time2-t02
    dt=time1[0]-time1[1]
    
    I1=I1/np.sum(I1)*np.sum(Ic1all)
    I2=I2/np.sum(I2)*np.sum(Ic1all2)
    
    q1=np.sum(I1)*dt*10**(-3)
    q2=np.sum(I2)*dt*10**(-3)
    print('Total q (pC):',np.sum(I1)*dt*10**(-3))
    print('Total q (pC):',np.sum(I2)*dt*10**(-3))
    
    
    
    s1=14
    s2=16
    
    f = plt.figure()
    trms=sigmafit(time1,I1*10**(-3))[2]
    trms=np.int(np.round(trms*10))/10
    trms1=sigmafit(time2,I2*10**(-3))[2]
    trms1=np.int(np.round(trms1*10))/10
    plt.plot(-time1,I1*10**(-3),'k',label='1st: '+str(trms)+' fs')
    plt.plot(time2,I2*10**(-3),'r',label='2nd+: '+str(trms1)+' fs')
    plt.plot(-time2,I2*10**(-3),'g:',label='2nd-: '+str(trms1)+' fs')
    plt.legend(fontsize=s1-3)
    plt.xlim(-100, 130)
    plt.xlabel('time (fs)',fontsize=s1) 
    plt.ylabel('current (kA)',fontsize=s1) 
    plt.xticks(fontsize=s1)
    plt.yticks(fontsize=s1)
    #plt.plot(time1,Ifit1)
    plt.title('TDC measurements (two zero crossings, rms)',fontsize=s2)
    f.savefig("TDC_Oct3.png", bbox_inches='tight')
    
   
    #plt.xticks(fontsize=s1)
    #plt.yticks(fontsize=s1)
#plt.legend(fontsize=s1)
##plt.plot(time1,Ifit1)
#plt.title('Streaked profile, 260 \u03BCm',fontsize=s2)
    plt.show()

    print('calibration factors:',cf1,cf2)
    print('ratio: ',cf1/cf2)
    return time1,I1,time2,I2,q1,q2


#def gaus2G(x,a1,x01,sigma1,a2,x02,sigma2,a3,x03,sigma3,a4,x04,sigma4):
#    f1=a1**2*exp(-(x-x01)**2/(2*sigma1**2))/m.sqrt(2*m.pi)/sigma1
#    f2=a2**2*exp(-(x-x02)**2/(2*sigma2**2))/m.sqrt(2*m.pi)/sigma2
#    f3=a3**2*exp(-(x-x03)**2/(2*sigma3**2))/m.sqrt(2*m.pi)/sigma3
#    f4=a4**2*exp(-(x-x04)**2/(2*sigma4**2))/m.sqrt(2*m.pi)/sigma4
#    return (f1+f2+f3+f4)


def gausnG(x,*params):
    n=np.int(len(params)/3)
    a=np.empty(n,dtype=float)
    x0=np.empty(n,dtype=float)
    sigma=np.empty(n,dtype=float)
    for i in range(n):
        a[i]=params[3*i]
        x0[i]=params[3*i+1]
        sigma[i]=params[3*i+2]
    f=0
    for i in range(n):
        f=f+a[i]**2*exp(-(x-x0[i])**2/(2*sigma[i]**2))/m.sqrt(2*m.pi)/sigma[i]
    return f

#def sigmafit2G(x,y):
#    mean = sum(x*y)/np.sum(y)
#    sigma = m.sqrt(abs(sum(y*(x-mean)**2)/np.sum(y)))  
#    popt,pcov = curve_fit(gaus2G,x,y,p0=[1,50,10,1,-50,10,1,0,10,1,10,10],maxfev=2000)
#    return popt


def sigmafitnG(x,y,g):
    #the number of gaussian modes is defined by guess
    popt,pcov = curve_fit(gausnG,x,y,p0=g,maxfev=2000)
    return popt


def fitTDC(t,I,g):
    dt=abs(t[1]-t[2])
    print('dt',dt)
    aout=np.copy(sigmafitnG(t,I/np.sum(I)/dt,g))
    #print(aout)
    plt.plot(t,I*10**(-3),'g',label='TDC measurements fit')
    plt.plot(t,gausnG(t,*aout)*np.sum(I)*dt*10**(-3),'r:',label=' MG fit')
    plt.legend()
    plt.show()
    return aout


def makeZ(Np,step):
    x=np.array(range(Np))*step-Np/2*step
    return x


def distMz(x,gP):
    return gausnG(x/0.30,*gP)/0.30

def wake(s,s0yd):
    return 1-(1+m.sqrt(s/s0yd))*m.exp(-m.sqrt(s/s0yd))

def w0(dy,a):
    c=3*10**8
    Z0=376.73
    return Q/En*L*Z0*c/4*m.pi**2/(4*a**3)*m.cos(m.pi*(a-dy)/2/a)**(-2)*m.tan(m.pi*(a-dy)/2/a)

#________________________________________
def modelAct(offset,L,En,Q,Npz,dz,gP):
    s0yd=16/9*250/(2*m.pi*0.6362**2)*offset**2/(500**2)
    def f2(s,s0,s0yd):
        f=distMz(s-s0,gP)*wake(s0,s0yd)*w0(offset*10**(-6),10**(-3))
        return f

    def I2(s,s0yd):
        return integrate.quad(lambda s0: f2(s,s0,s0yd), 0, +m.inf)

    z=makeZ(Npz,dz)
    
    IA=distMz(z,gP)
    
   
    y=np.empty(Npz,dtype=float)
    for i in range(0,len(z)):
        y[i]=s0yd*I2(z[i],s0yd)[0]

    return z,y,IA   

def modelRec(y,y1,Nint,yp,dz):
    #y nonlinearly sampled points
    #y1 function value on these points
    prts=np.empty(0)
    for i in range(1,len(y)-1):   
        xb=(y[i]-y[i-1])/2+y[i]
        dxi=(y[i+1]-y[i])/2+(y[i]-y[i-1])/2
        Ni=np.int(np.round(y1[i]*dz*Nint))
        prtsI=np.empty(Ni,dtype=float)
        for j in range(Ni):
            prtsI[j]=xb+random()*dxi
        
        prts=np.concatenate((prts, prtsI), axis=None)
    
    print('particles lost:',Nint-len(prts))

    ymax=np.max(y)
    ymin=np.min(y)
   
    Nd=np.int(np.round((ymax-ymin)/yp))
    

    yD=np.empty(Nd,dtype=float)
    for i in range(len(yD)): 
        yD[i]=ymin+i*yp


    distD=np.empty(len(yD),dtype=float)
    for i in range(1,len(yD)-1):   
        xb=(yD[i]-yD[i-1])/2+yD[i]
        dxi=(yD[i+1]-yD[i])/2+(yD[i]-yD[i-1])/2
        xe=xb+dxi
        distD[i]=np.count_nonzero((prts < xe)&(prts > xb))

    distD[-1]=0  
    distD[0]=0 
    distD=distD/(dz*Nint)
    #adding 0 for the good fit
    yDe=np.empty(2*len(yD),dtype=float)
    Ne=np.empty(2*len(yD),dtype=float)

    yDe[len(yD):2*len(yD)]=yD
    Ne[len(yD):2*len(yD)]=distD/(dz*Nint)

    for i in range(len(yD)):
        yDe[i]=-yD[len(yD)-1-i]
        Ne[i]=0.0
        
  
    return yD,distD,sigmafit(yDe,Ne)[2],np.sum(Ne)



def addnaturalsize(xA,fA,sigmaX,N):
    
    NpE=N+len(xA)


    dxA=xA[1]-xA[0]

    xAe=np.empty(NpE,dtype=float)
    fAe=np.empty(NpE,dtype=float)


    xAe=(np.array(range(NpE),dtype=float)-N)*dxA
    for i in range(NpE-N):
        fAe[i]=0
    
    fAe[N:NpE]=np.copy(fA)

    fAS=np.zeros(NpE,dtype=float)
    for i in range(NpE):
        ns=np.empty(NpE,dtype=float)
        for j in range(NpE):
            ns[j]=distS(xAe[j],sigmaX,xAe[i])
        fAS=fAS+ns*fAe[i]*dxA
        
    ##trick here take the particles before 0 and add them app to the distribution

    ind=np.array(range(N))
    df=fAe-fAS

    fReM=np.copy(fAS)
    fReM[N+1:2*N+1]=fReM[N+1:2*N+1]-df[N-ind]

 
    return xA,fAS[N:NpE],xAe,fAS,fAe,fReM[N:NpE]
  

        





#______________________________code
    
outTDC=readTDC(filenameTDC)
##
##
t=outTDC[0]
###
###
#g0=[1,50,10,1,-50,10,1,0,10,1,10,10,1,-10,10]
g0=[1,50,10,1,-50,10,1,1,10,1,10,10,1,-10,10]
##gP=fitTDC(outTDC[2],outTDC[3],g0)
gP=fitTDC(-outTDC[0][500:1000],outTDC[1][500:1000],g0)

##
##
###plt.plot(t,gausnG(t,*gP))
###plt.xlabel('t (fs)') 
###plt.ylabel('current (a.u.)') 
###plt.show()
##
x=makeZ(1600,0.05)
#
#
#plt.plot(t*0.3,gausnG(t,*gP)/0.3)
#plt.plot(x,distMz(x,gP))


#plt.xlabel('z (um)') 
#plt.ylabel('current (a.u.)') 
#plt.show()
##
###
###
###
###
#Q=200.6*10**(-12)
##En=6.2*10**9 #6 GeV @ July 26
#En=4.5*10**9 #4.5 GeV @ Oct 3
##L=15.5 #July 26
#L=5.957 #Octob 3
#Npz=1500
#dz=0.05
#Nint=1000000
#yp=8.737
##
##
##
##    
#off=[310,300,350]
#
#wakeoutAct=modelAct(off[0],L,En,Q,Npz,dz,gP)
#zA=wakeoutAct[0]
#yA=wakeoutAct[1]
#xA=wakeoutAct[2]
#
#plt.plot(zA,yA)
#plt.show()
#
##
#ScreenOutAct=modelRec(yA,xA,Nint,yp,dz)
#
#xScA=ScreenOutAct[0]
#IscA=ScreenOutAct[1]
#
#plt.plot(xScA,IscA,'g',label='210 um, Act+')
#plt.show()
#
#N=100
#sig=5.94479#JULY 26
#sig=3.96765
#sigmaX=yp*sig
##sigmaX=30
#
#outSA=addnaturalsize(xScA,IscA,sigmaX,N)
#
#plt.plot(outSA[2],outSA[3],'r',label='210 um, Act+')
#plt.plot(xScA,IscA,'g:',label='210 um, Act+')
#plt.show()

#
##
##
##
##
#D3=['SpectrumAnalysis_2020_07_26_17_56_25_401050.h5','SpectrumAnalysis_2020_07_26_17_55_28_676729.h5',
#    'SpectrumAnalysis_2020_07_26_17_46_52_221317.h5',
#    'SpectrumAnalysis_2020_07_26_17_45_20_885525.h5','SpectrumAnalysis_2020_07_26_17_41_01_350920.h5',
#    'SpectrumAnalysis_2020_07_26_17_39_37_960314.h5','SpectrumAnalysis_2020_07_26_17_34_33_740151.h5',
#    ]
#fPS3 = h5py.File(D3[4], 'r')
#fPS3A = h5py.File(D3[5], 'r')
#fPS0 = h5py.File(D3[0], 'r')
#abcA=np.array(fPS3A['scan_1']['data']['SARBD02-DSCR050']['FPICTURE'])
#bcA=abcA.reshape(abcA.shape[0]*abcA.shape[1],abcA.shape[2],abcA.shape[3])
#abc=np.array(fPS3['scan_1']['data']['SARBD02-DSCR050']['FPICTURE'])
#bc=abc.reshape(abc.shape[0]*abc.shape[1],abc.shape[2],abc.shape[3])
#abc0=np.array(fPS0['scan_1']['data']['SARBD02-DSCR050']['FPICTURE'])
#bc0=abc0.reshape(abc0.shape[0]*abc0.shape[1],abc0.shape[2],abc0.shape[3])
#PS3=bc[0]
#PS3A=bcA[0]
#PS0=bc0[0]
#I31A=np.sum(PS3A,axis=0)-np.mean(np.sum(PS3A,axis=0)[0:10])
#I31=np.sum(PS3,axis=0)-np.mean(np.sum(PS3,axis=0)[0:10])
#I01=np.sum(PS0,axis=0)-np.mean(np.sum(PS0,axis=0)[0:10])
#PS3A=bcA[11]
#I32A=np.sum(PS3A,axis=0)-np.mean(np.sum(PS3A,axis=0)[0:10])
#PS3=bc[11]
#PS0=bc0[1]
#I32=np.sum(PS3,axis=0)-np.mean(np.sum(PS3,axis=0)[0:10])
#I02=np.sum(PS0,axis=0)-np.mean(np.sum(PS0,axis=0)[0:10])
#
#Ns=len(bc0)
#Nx=np.array(range(len(I02)),dtype=float)
#print(sigmafit(Nx,I02))
#
##plt.plot(I02,label='sample2, nbs')
##plt.plot(Nx,gaus(Nx,*sigmafit(Nx,I02)),label='sample2 - fit')
##plt.legend()
##plt.show()
#
#s0=np.empty(Ns,dtype=float)
#sd0=np.empty(Ns,dtype=float)
#for i in range(Ns):
#    PS0=bc0[i]
#    I0=np.sum(PS0,axis=0)-np.mean(np.sum(PS0,axis=0)[0:10])
#    s0[i]=sigmafit(Nx,I0)[1]
#    sd0[i]=np.abs(sigmafit(Nx,I0)[2])
#    
#print('mean 0 pixel',np.mean(s0))
#print('std 0 pixel',np.std(s0))
#
#print('mean 0 pixel',np.mean(sd0))
#print('std 0 pixel',np.std(sd0))
##    
#    
#p0=np.int(np.mean(s0)) #center-of-mass of the unstreaked beam!!!
#
##plt.plot(I31[p0:1600],label='sample1, 210 um')
###plt.plot(I32[1000:1600],label='sample2, 210 um')
###plt.plot(I31A[1000:1600],label='sample1, 260 um')
###plt.plot(I32A[1000:1600],label='sample2, 260 um')
###plt.plot(I01[1000:1600],label='sample1, nbs ')
###plt.plot(I02[1000:1600],label='sample2, nbs')
##plt.legend()
##plt.show()
#
#yp=8.737
###
#N1=1050
#Np=560
#N2=N1+Np
#xSc=(np.array(range(Np))-p0+N1)*yp
#Im=I31[N1:N2]-np.mean(I31[1600:1700])
#Im1=I32[N1:N2]-np.mean(I32[1600:1700])
#
#s1=14
#s2=16
#
#
#
#f = plt.figure()
#plt.plot(-xSc,I2[N1:N2]/np.sum(I2)*50,'b:',label='measured')
##plt.plot(xSc,Im1/np.sum(Im1),'g')
##plt.plot(x250m,y250m,'g',label='TDC propagated 2nd-')
#plt.plot(outSA[2][115:675],outSA[3][115:675]/np.sum(outSA[3])*50,'r',label='TDC propagated 2nd+')
#
#plt.xlabel('position x (\u03BCm)',fontsize=s1) 
#plt.ylabel('density (arb. units)',fontsize=s1)
#plt.xticks(fontsize=s1)
#plt.yticks(fontsize=s1)
#plt.legend(fontsize=s1)
##plt.plot(time1,Ifit1)
#plt.title('Streaked profile, 250 \u03BCm',fontsize=s2)
#
#plt.show()
###
#f.savefig("Oct_280um_both.png", bbox_inches='tight')
#
#
#f = plt.figure()
#plt.plot(xSc,Im/np.sum(Im)*50,'b:',label='measured')
##plt.plot(xSc,Im1/np.sum(Im1),'g')
#plt.plot(x300m,y300m,'g',label='TDC propagated 1st')
#plt.plot(outSA[2][115:475],outSA[3][115:475]/np.sum(outSA[3])*50,'r',label='TDC propagated 2nd')
#plt.xlabel('position x (\u03BCm)',fontsize=s1) 
#plt.ylabel('density (arb. units)',fontsize=s1)
#plt.xticks(fontsize=s1)
#plt.yticks(fontsize=s1)
#plt.legend(fontsize=s1)
##plt.plot(time1,Ifit1)
#plt.title('Streaked profile, 300 \u03BCm',fontsize=s2)
#
#plt.show()
#
#f.savefig("300um_both.png", bbox_inches='tight')
#
#
#
#
#
#f = plt.figure()
#plt.plot(xSc,Im/np.sum(Im)*50,'b:',label='measured')
##plt.plot(xSc,Im1/np.sum(Im1),'g')
#plt.plot(x350m,y350m,'g',label='TDC propagated 2nd-')
##plt.plot(x310,y310,'m',label='TDC propagated 2nd-old')
#plt.plot(outSA[2][115:415],outSA[3][115:415]/np.sum(outSA[3])*50,'r',label='TDC propagated 2nd+')
#plt.xlabel('position x (\u03BCm)',fontsize=s1) 
#plt.ylabel('density (arb. units)',fontsize=s1)
#plt.xticks(fontsize=s1)
#plt.yticks(fontsize=s1)
#plt.legend(fontsize=s1)
##plt.plot(time1,Ifit1)
#plt.title('Streaked profile, 350 \u03BCm',fontsize=s2)
#
#plt.show()
#
#f.savefig("310um_both_Oct_40fs.png", bbox_inches='tight')


##








##plt.plot(t,gausnG(t,a,x0,sigma))
##plt.show()
#
#I=outTDC[1]
#
#
#
#
#
#p0=[[1,1,1,1],[50,-50,-10,10],[10,10,10,10]]
#
#plt.plot(t,gausnG(t,*p0))
#plt.show()
#
#a=['a1','a2','a3']
#
#print(a)


#print(sigmafitnG(outTDC[0],I/np.sum(I)))



