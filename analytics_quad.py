#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:04:14 2021

@author: alex
"""
#Mathematica file: "quad wake estimates" for matrix formalism
def emittanceEstN(R11,R12,Alpha,Beta,Gamma,NBS,gbeam):
    return NBS**2/(-2*R11*R12*Alpha+R11**2*Beta+R12**2*Gamma)*gbeam

def quadsize(R11,R12,Alpha,Beta,Gamma,gbeam,emitNx,QuadKick):
    return (QuadKick*R12*(-2*R12*Alpha+2*R11*Beta+QuadKick*R12*Beta)*emitNx/gbeam)**0.5
    

#transport matrix from Elegant (Philipp)
R11x03 = -6.55652876*10**(-1)
R12x03 = 5.95682563
R21x03 = -1.33860414*10**(-2)
R22x03 = -1.40358064

Ls=1


R12x03S = R12x03+R11x03*Ls #including streaker length in matrix

R11x04 = -8.56522444*10**(-1)
R12x04 = 8.27920255
R21x04 = -2.07952217*10**(-1)
R22x04 = 8.42568143*10**(-1)

BetaX = 11.110349832962324
AlphaX = -1.39472106229871 #Beta*Gamma-Alpha^2=1
GammaX=(1+AlphaX**2)/BetaX

Energy03=4491.8929157690#MeV
Energy04=4480.908561 #MeV

sigmaX = 32.37

sigmaX03=sigmaX*10**(-6)#from  the measurements 03_Oct
sigmaX04=43*10**(-6) ##???


enx03=emittanceEstN(R11x03,R12x03,AlphaX,BetaX,GammaX,sigmaX03,Energy03/0.511)
print('Oct 03 (long):estimated normalized emittance in x [nm]=',enx03*10**9)

enx03=emittanceEstN(R11x03,R12x03S,AlphaX,BetaX,GammaX,sigmaX03,Energy03/0.511)
print('Oct 03 (long):estimated normalized emittance in x, +streaker length [nm]=',enx03*10**9)

enx04=emittanceEstN(R11x04,R12x04,AlphaX,BetaX,GammaX,sigmaX04,Energy04/0.511)
print('Oct 04 (short):estimated normalized emittance in x [nm]=',enx04*10**9)

QS=quadsize(R11x03,R12x03,AlphaX,BetaX,GammaX,Energy03/0.511,enx03,0.1)
print('Quad component (const kick assumption) [um]=',QS*10**6)

    
