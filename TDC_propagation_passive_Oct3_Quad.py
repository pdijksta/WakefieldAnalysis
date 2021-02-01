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


filenameTDC = 'Bunch_length_meas_2020-07-26_15-03-59.h5' #July26

#filenameTDC ='129858802_bunch_length_meas.h5' #October 4 night

filenameTDC ='129833611_bunch_length_meas.h5' #October 4 night


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
    #print('dt=',dt)
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
    #undimensional
    return 1-(1+m.sqrt(s/s0yd))*m.exp(-m.sqrt(s/s0yd))



def w0(dy,a):
    # e/eV*m*(V*s/C)*m/s/m^3
    c=3*10**8
    Z0=376.73
    return Q/En*L*Z0*c/4*m.pi**2/(4*a**3)*m.cos(m.pi*(a-dy)/2/a)**(-2)*m.tan(m.pi*(a-dy)/2/a)

def w0Q(dy,a):
    # e/eV*m*(V*s/C)*m/s/m^3
    c=3*10**8
    Z0=376.73
    return Q/En*L*Z0*c/4*m.pi**3/(16*a**4)*m.cos(m.pi*(a-dy)/2/a)**(-4)*(2-m.cos(m.pi*(a-dy)/a))

#________________________________________
    
def modelSingle(offset,L,En,Q,Npz,dz,gP):
    s0yd=16/9*250/(2*m.pi*0.6362**2)*offset**2/(500**2)
    def fS(s0,s0yd):
        f=wake(s0,s0yd)*w0(offset*10**(-6),10**(-3))
        return f
    
def modelSingleF(offset,L,En,Q,Npz,dz,gP,dy,a):
    s0yd=16/9*250/(2*m.pi*0.6362**2)*offset**2/(500**2)
   # s0yd=4*s0yr*(3/2+m.pi*(a-dy)/a*(m.sin(m.pi*(a-dy)/a))**(-1)-m.pi*(a-dy)/2/a*(m.tan(m.pi*(a-dy)/a))**(-1))**-2
    def fS(s0,s0yd):
        f=wake(s0,s0yd)*w0(offset*10**(-6),10**(-3))
        return f


    z=np.array(range(Npz))*dz
    
    
   
    y=np.empty(Npz,dtype=float)
    for i in range(0,len(z)):
        y[i]=s0yd*fS(z[i],s0yd)

    return z,y





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


def modelActF(offset,L,En,Q,Npz,dz,gP):
    a=10**(-3)#half gap
    dy=offset*10**(-6)
    tc=250*10**(-6)#period of the corrugation
    pc=500*10**(-6)#longitudinal gap of the corrugation
    alpha=0.6362
    s0yr=a**2*tc/(2*m.pi*pc**2*alpha**2)
  
    s0yd=4*s0yr*(3/2+m.pi*(a-dy)/a*(m.sin(m.pi*(a-dy)/a))**(-1)-m.pi*(a-dy)/2/a*(m.tan(m.pi*(a-dy)/a))**(-1))**-2*10**(6)
    
   # s0yd=16/9*250/(2*m.pi*0.6362**2)*offset**2/(500**2)
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

#20 um   
    

def modelActQ(offset,L,En,Q,Npz,dz,gP,ybeam):
    a=10**(-3)#half gap
    dy=offset*10**(-6)
    tc=250*10**(-6)#period of the corrugation
    pc=500*10**(-6)#longitudinal gap of the corrugation
    alpha=0.6362
    s0yr=a**2*tc/(2*m.pi*pc**2*alpha**2)
  
    s0yd=4*s0yr*(3/2+m.pi*(a-dy)/a*(m.sin(m.pi*(a-dy)/a))**(-1)-m.pi*(a-dy)/2/a*(m.tan(m.pi*(a-dy)/a))**(-1))**-2*10**(6)
    
    Th=m.pi*(a-dy)/2/a
    s0yq=4*s0yr*((56-m.cos(2*Th))/30+(0.3+Th*m.sin(2*Th))/(2-m.cos(2*Th))+2*Th*m.tan(Th))**(-2)*10**(6)
    
    
   # s0yd=16/9*250/(2*m.pi*0.6362**2)*offset**2/(500**2)
    def f2(s,s0,s0yd):
        f=distMz(s-s0,gP)*wake(s0,s0yd)*w0(offset*10**(-6),10**(-3))
        return f

    def I2(s,s0yd):
        return integrate.quad(lambda s0: f2(s,s0,s0yd), 0, +m.inf)
    
    def f2Q(s,s0,s0yq):
        f=distMz(s-s0,gP)*wake(s0,s0yq)*w0Q(offset*10**(-6),10**(-3))
        return f

    def I2Q(s,s0yq):
        return integrate.quad(lambda s0: f2Q(s,s0,s0yq), 0, +m.inf)

    z=makeZ(Npz,dz)
    
    IA=distMz(z,gP)
    
   
    y=np.empty(Npz,dtype=float)
    for i in range(0,len(z)):
        y[i]=s0yd*I2(z[i],s0yd)[0]
    
    yQ=np.empty(Npz,dtype=float)
    for i in range(0,len(z)):
        yQ[i]=ybeam*s0yq*I2Q(z[i],s0yq)[0]

    return z,y,IA,yQ



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
####
####
#t=outTDC[2]
#####
#####
#g0=[1,50,10,1,-50,10,1,0,10,1,10,10,1,-10,10]
g0=[1,50,10,1,-50,10,1,1,10,1,10,10,1,-10,10]
#gP=fitTDC(outTDC[2],outTDC[3],g0)
print('-------------------------------')
print('fitting the TDC measurements with multi-gaussain model:')
gP=fitTDC(-outTDC[0][500:1000],outTDC[1][500:1000],g0)
###
###
####plt.plot(t,gausnG(t,*gP))
####plt.xlabel('t (fs)') 
####plt.ylabel('current (a.u.)') 
####plt.show()
###
#x=makeZ(1500,0.05)
##
##
#plt.plot(t*0.3,gausnG(t,*gP)/0.3)
#plt.plot(x,distMz(x,gP))
#
#
#plt.xlabel('z (um)') 
#plt.ylabel('current (a.u.)') 
#plt.show()
###
####
####
####
####
Q=200.6*10**(-12)
#En=6.2*10**9 #6 GeV @ July 26
En=4.5*10**9 #4.5 GeV @ Oct 3
#L=15.5 #July 26
L=5.957 #Octob 3
Npz=1500
dz=0.05
Nint=1000000
yp=8.737 #pixel size on the screen in um
#
#
#
#    
off=[310,300,350]
#


#nW=26
#off0=230
#yAall=np.empty((nW,Npz),dtype=float)
#offsets=np.empty((nW),dtype=float)
#for i in range(nW):
#    offS=off0+i*5
#    offsets[i]=offS
#    wakeoutAct=modelAct(offS,L,En,Q,Npz,dz,gP)
#    yAall[i]=wakeoutAct[1]
   

wakeoutAct=modelAct(off[0],L,En,Q,Npz,dz,gP)
zA=wakeoutAct[0]
yA=wakeoutAct[1]
xA=wakeoutAct[2]


wakeoutActP=modelAct(off[0]-20,L,En,Q,Npz,dz,gP)
zAP=wakeoutActP[0]
yAP=wakeoutActP[1]
xAP=wakeoutActP[2]


wakeoutActM=modelAct(off[0]+20,L,En,Q,Npz,dz,gP)
zAM=wakeoutActM[0]
yAM=wakeoutActM[1]
xAM=wakeoutActM[2]


ybeam1=20*10**(-6)
wakeoutActQ=modelActQ(off[0],L,En,Q,Npz,dz,gP,ybeam1)
zAF=wakeoutActQ[0]
yAF=wakeoutActQ[1]
xAF=wakeoutActQ[2]
yAFQ=wakeoutActQ[3]

s1=14
s2=16

print('-------------------------------')
print('the nonlinear correspondence function, x(t) {dipole kick}:')
plt.plot(zA/0.3,yA,label='approx. dipole')
plt.plot(zAF/0.3,yAF,label='exact dipole')
plt.plot(zAF/0.3,yAFQ,label='exact quad')
plt.xlabel('t (fs)',fontsize=s1) 
plt.ylabel('x (um)',fontsize=s1) 
plt.xticks(fontsize=s1)
plt.yticks(fontsize=s1)
plt.legend()
    #plt.plot(time1,Ifit1)
#plt.title('TDC measurements (two zero crossings, rms)',fontsize=s2)
plt.show()

#
ScreenOutAct=modelRec(yA,xA,Nint,yp,dz)

xScA=ScreenOutAct[0]
IscA=ScreenOutAct[1]

print('-------------------------------')
print('the simulated image on the screen (assuming 0 natural beam size, but fixed pixel size):')
plt.plot(xScA,IscA,'g',label='210 um, Act+')
plt.ylabel('density (arb. units)',fontsize=s1) 
plt.xlabel('x (um)',fontsize=s1) 
plt.xticks(fontsize=s1)
plt.yticks(fontsize=s1)
plt.show()

N=200
#sig=5.94479#JULY 26
sig=3.977269  #rms bunch size of the unstreaked beam in pixels
sigmaX=yp*sig #rms bunch size of the unstreaked beam in um
#sigmaX=5

print('-------------------------------')
print('the simulated image on the screen (after convolution with natural beam size):')
outSA=addnaturalsize(xScA,IscA,sigmaX,N)

plt.plot(outSA[2],outSA[3],'k',label='310 um, Act+')
plt.xlabel('density (arb. units)',fontsize=s1) 
plt.ylabel('x (um)',fontsize=s1) 
plt.xticks(fontsize=s1)
plt.yticks(fontsize=s1)
plt.legend()

#plt.plot(xA1,IA1,'r',label='310 um, Act+') #measured on the screen
#plt.plot(xScA,IscA,'g:',label='210 um, Act+')
plt.show()



#print('-------------------------------')
#print('resolution:')


#nW=26
#off0=230
#yAall=np.empty((nW,Npz),dtype=float)
#offsets=np.empty((nW),dtype=float)
#for i in range(nW):
#    offS=off0+i*5
#    offsets[i]=offS
#    wakeoutAct=modelAct(offS,L,En,Q,Npz,dz,gP)
#    yAall[i]=wakeoutAct[1]
#    
#f0 = plt.figure()   
#for i in range(8):
#    
#    plt.plot(zA/0.3,yAall[i*2],label=str(offsets[2*i])+' \u03BCm')
#    
#
#plt.plot(t,gausnG(t,*gP)*400000,'k:',label='profile')
#plt.legend(fontsize=s1-2)
#plt.xlim(-150, 130)
#plt.xlabel('time (fs)',fontsize=s1) 
#plt.ylabel('x (\u03BCm)',fontsize=s1) 
#plt.xticks(fontsize=s1)
#plt.yticks(fontsize=s1)
#    
#plt.title('Convoluted wakefield (different offsets)',fontsize=s2)
#f0.savefig("Convoluted_wake.png", bbox_inches='tight')
#plt.show()
#  
    


#xS330=outSA[2]
#IS330=outSA[3]


#print('how to overcome pixel resolution in the first horn?')

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
#fPS3 = h5py.File(D3[6], 'r')
#fPS3A = h5py.File(D3[5], 'r')
#fPS3B = h5py.File(D3[4], 'r')
#fPS0 = h5py.File(D3[0], 'r')
#abcA=np.array(fPS3A['scan_1']['data']['SARBD02-DSCR050']['FPICTURE'])
#bcA=abcA.reshape(abcA.shape[0]*abcA.shape[1],abcA.shape[2],abcA.shape[3])
#abcB=np.array(fPS3B['scan_1']['data']['SARBD02-DSCR050']['FPICTURE'])
#bcB=abcB.reshape(abcB.shape[0]*abcB.shape[1],abcB.shape[2],abcB.shape[3])
#
#abc=np.array(fPS3['scan_1']['data']['SARBD02-DSCR050']['FPICTURE'])
#bc=abc.reshape(abc.shape[0]*abc.shape[1],abc.shape[2],abc.shape[3])
#abc0=np.array(fPS0['scan_1']['data']['SARBD02-DSCR050']['FPICTURE'])
#bc0=abc0.reshape(abc0.shape[0]*abc0.shape[1],abc0.shape[2],abc0.shape[3])
#PS3=bc[15]
#PS3A=bcA[0]
#PS0=bc0[0]
#I31A=np.sum(PS3A,axis=0)-np.mean(np.sum(PS3A,axis=0)[0:10])
#I31=np.sum(PS3,axis=0)-np.mean(np.sum(PS3,axis=0)[0:10])
#
#n=len(bc)
#I31All=np.empty((n,len(I31)), dtype=float)
#for i in range(n):
#    I31All[i]=np.sum(bc[i],axis=0)-np.mean(np.sum(bc[i],axis=0)[0:10])
#I31av=np.mean(I31All,axis=0)
#I31a=np.std(I31All,axis=0)
#
#
#
#I32All=np.empty((n,len(I31)), dtype=float)
#for i in range(n):
#    I32All[i]=np.sum(bcA[i],axis=0)-np.mean(np.sum(bcA[i],axis=0)[0:10])
#I32av=np.mean(I32All,axis=0)
#I32a=np.std(I32All,axis=0)
#
#
#I33All=np.empty((n,len(I31)), dtype=float)
#for i in range(n):
#    I33All[i]=np.sum(bcB[i],axis=0)-np.mean(np.sum(bcB[i],axis=0)[0:10])
#I33av=np.mean(I33All,axis=0)
#I33a=np.std(I33All,axis=0)
#
#
##
##
##
##
##
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
#plt.plot(I02,label='sample2, nbs')
#plt.plot(Nx,gaus(Nx,*sigmafit(Nx,I02)),label='sample2 - fit')
#plt.legend()
#plt.show()

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
###    
##    
#p0=np.int(np.mean(s0)) #center-of-mass of the unstreaked beam!!!
##
###plt.plot(I31[p0:1600],label='sample1, 210 um')
####plt.plot(I32[1000:1600],label='sample2, 210 um')
####plt.plot(I31A[1000:1600],label='sample1, 260 um')
####plt.plot(I32A[1000:1600],label='sample2, 260 um')
####plt.plot(I01[1000:1600],label='sample1, nbs ')
####plt.plot(I02[1000:1600],label='sample2, nbs')
###plt.legend()
###plt.show()
##
#yp=8.737
###
#N1=1050
#Np=560
#N2=N1+Np
#xSc=(np.array(range(Np))-p0+N1)*yp
#Im1=I31av[N1:N2]-np.mean(I31av[1600:1700])
#Im1S=I31a[N1:N2]-np.mean(I31a[1600:1700])
#
#Im2=I32av[N1:N2]-np.mean(I32av[1600:1700])
#Im2S=I32a[N1:N2]-np.mean(I32a[1600:1700])
#
#Im3=I33av[N1:N2]-np.mean(I33av[1600:1700])
#Im3S=I33a[N1:N2]-np.mean(I33a[1600:1700])
#
#s1=14
#s2=16

#
#
#f = plt.figure()
#plt.plot(xSc,Im/np.sum(Im1)*50,'b',label='Measured average')
#plt.fill_between(xSc,(Im-ImS)/np.sum(Im1)*50,(Im+ImS)/np.sum(Im1)*50)
#plt.plot(xSc,Im1/np.sum(Im1)*50,'g',label='Measured')
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
#
#
#
#
#plt.plot(xSc,Im2/np.sum(Im2)*50,'b',label='Measured average')
#plt.fill_between(xSc,(Im2-Im2S)/np.sum(Im2)*50,(Im2+Im2S)/np.sum(Im2)*50)
#
##plt.plot(x250m,y250m,'g',label='TDC propagated 2nd-')
##plt.plot(outSA[2][115:675],outSA[3][115:675]/np.sum(outSA[3])*50,'r',label='TDC propagated 2nd+')
#
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
#
#
#plt.plot(xSc,Im3/np.sum(Im3)*50,'b',label='Measured average')
#plt.fill_between(xSc,(Im3-Im3S)/np.sum(Im3)*50,(Im3+Im3S)/np.sum(Im3)*50)
#
##plt.plot(x250m,y250m,'g',label='TDC propagated 2nd-')
##plt.plot(outSA[2][115:675],outSA[3][115:675]/np.sum(outSA[3])*50,'r',label='TDC propagated 2nd+')
#
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
#
#
#plt.plot(xSc,Im3/np.sum(Im3)*50,'g',label='Measured average, 350 \u03BCm')
#plt.fill_between(xSc,(Im3-Im3S)/np.sum(Im3)*50,(Im3+Im3S)/np.sum(Im3)*50)
##plt.plot(xSc,Im2/np.sum(Im2)*50,'b',label='Measured average, 300 \u03BCm')
##plt.fill_between(xSc,(Im2-Im2S)/np.sum(Im2)*50,(Im2+Im2S)/np.sum(Im2)*50)
#
#plt.plot(xSc,Im1/np.sum(Im1)*50,'m',label='Measured average, 250 \u03BCm')
#
#
#plt.fill_between(xSc,(Im1-ImS)/np.sum(Im1)*50,(Im1+ImS)/np.sum(Im1)*50)
#
#
##plt.plot(x250m,y250m,'g',label='TDC propagated 2nd-')
##plt.plot(outSA[2][115:675],outSA[3][115:675]/np.sum(outSA[3])*50,'r',label='TDC propagated 2nd+')
#
#plt.xlabel('position x (\u03BCm)',fontsize=s1) 
#plt.ylabel('density (arb. units)',fontsize=s1)
#plt.xticks(fontsize=s1)
#plt.yticks(fontsize=s1)
#plt.legend(fontsize=s1-2)
##plt.plot(time1,Ifit1)
#plt.title('Streaked profile, 350 \u03BCm',fontsize=s2)
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



