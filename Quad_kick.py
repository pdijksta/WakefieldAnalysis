#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:56:32 2021

@author: alex
"""

print('the nonlinear correspondence function, x(t) {dipole and quad kick}:')
#plt.plot(zA,yA)
plt.plot(zAF/0.3,yAF,label='0 um (dipole only)')
plt.plot(zAF/0.3,yAF+yAFQ,label='+20 um particle (d+q)')
plt.plot(zAF/0.3,yAF-yAFQ,label='-20 um particle (d+q)')
plt.xlabel('t (fs)',fontsize=s1) 
plt.ylabel('x (um)',fontsize=s1) 
plt.xticks(fontsize=s1)
plt.yticks(fontsize=s1)
plt.legend(fontsize=s1-2)
    #plt.plot(time1,Ifit1)
#plt.title('TDC measurements (two zero crossings, rms)',fontsize=s2)
plt.show()


print('the nonlinear correspondence function, x(t) {dipole and quad kick}:')
#plt.plot(zA,yA)
plt.plot(zAF/0.3,yAF,label='0 um (dipole only)')
plt.plot(zAF/0.3,yAF+yAFQ,label='+20 um particle (d+q)')
plt.plot(zAF/0.3,yAF-yAFQ,label='-20 um particle (d+q)')

plt.plot(zAM/0.3,yAM,label='-20 um (dipole only*)',linestyle=':')
plt.plot(zAP/0.3,yAP,label='+20 um (dipole only*)',linestyle=':')
plt.plot(zA/0.3,xA*13000,label='current profile (a.u.)',linestyle='-')
plt.xlabel('t (fs)',fontsize=s1) 
plt.ylabel('x (um)',fontsize=s1) 
plt.xlim(-70, 70)
plt.xticks(fontsize=s1)
plt.yticks(fontsize=s1)
plt.legend(fontsize=s1-2)
    #plt.plot(time1,Ifit1)
#plt.title('TDC measurements (two zero crossings, rms)',fontsize=s2)
plt.show()