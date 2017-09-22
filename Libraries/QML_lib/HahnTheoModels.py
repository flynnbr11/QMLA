import numpy as np
import scipy as sp
import qutip as qt
import Evo as evo
import sys as sys
import os as os


sys.path.append(os.path.join("..",".."))
import SETTINGS

"""
Functions to simulate the behaviour of an NV centre
under different possibile Hamiltonians
"""

## DEFINITIONS

spinlst = [evo.sigmax(), evo.sigmay(), evo.sigmaz()]    

eoplist = np.array(list(map(lambda j: np.kron(spinlst[j],np.eye(2)) , range(len(spinlst)))))
bathoplist = np.array(list(map(lambda j: np.kron(np.eye(2),spinlst[j]) , range(len(spinlst)))))
HFoplist = np.array(list(map(lambda j: np.kron(spinlst[j],spinlst[j]), range(len(spinlst)))))

probestate = np.kron(evo.plus(), evo.plus())

Hahn_angle = np.pi/2
H_Hahn = np.kron(Hahn_angle*spinlst[2],np.eye(2))

## 


def Q2_HF_Precession(tvec, HfA = [2.1, 2.1, 2.9], Bathfreq = [3, 0.87, 0.9]):

    evolve = np.empty(len(tvec))

    # Rabifreq = [1,1,1]

        
    """Dynamics on the nuclear spin""" 
    Htot =  np.tensordot(HfA, HFoplist, axes=1) +  np.tensordot(Bathfreq, bathoplist, axes=1)
    
    #Htot =  np.tensordot(HfA, HFoplist, axes=1) + np.tensordot(Rabifreq, eoplist, axes=1)
    
    for idt in range(len(tvec)):

        evostate = np.dot(sp.linalg.expm(-(1j)*1*H_Hahn), probestate)    # Hahn-echo operation

        evostate = np.dot(sp.linalg.expm(-(1j)*2*tvec[idt]*Htot), evostate)

        qt_evostate = qt.Qobj(evostate) # transforming into a Qutip object in order to compute partial trace
        qt_evostate.dims = [[2,2],[1,1]]

        ## Partial tracing over the nuclear deg of freedom
        
        evolve[idt] = 1-(qt.expect(qt_evostate.ptrace(0), qt.Qobj(evo.plus()) ))**2
        
    return evolve
    
    
def Q2_HF_Precession_deco(tvec, HfA = [2.1, 2.1, 2.9], Bathfreq = [3, 0.87, 0.9], decoTau_c = SETTINGS.decoTau_c):

    evolve = np.empty(len(tvec))

    # Rabifreq = [1,1,1]

        
    """Dynamics on the nuclear spin""" 
    Htot =  np.tensordot(HfA, HFoplist, axes=1) +  np.tensordot(Bathfreq, bathoplist, axes=1)
    
    #Htot =  np.tensordot(HfA, HFoplist, axes=1) + np.tensordot(Rabifreq, eoplist, axes=1)
    
    for idt in range(len(tvec)):

        evostate = np.dot(sp.linalg.expm(-(1j)*1*H_Hahn), probestate)    # Hahn-echo operation

        evostate = np.dot(sp.linalg.expm(-(1j)*2*tvec[idt]*Htot), evostate)

        qt_evostate = qt.Qobj(evostate) # transforming into a Qutip object in order to compute partial trace
        qt_evostate.dims = [[2,2],[1,1]]

        ## Partial tracing over the nuclear deg of freedom
        
        evolve[idt] = np.exp(-(2*tvec[idt]/decoTau_c)**4) * (1-(qt.expect(qt_evostate.ptrace(0), qt.Qobj(evo.plus()) ))**2) + (1-np.exp(-(2*tvec[idt]/decoTau_c)**4)) / 2
        
    return evolve
    
    
    
def OscBath_SignalModel(tvec, Nqubit, offset = 0., nrep = 10, freqpars = [0.385, 5.7*10**-2, 17*10**-3], Bpars= [0.45, 0.015], decoTau_c = SETTINGS.decoTau_c):

    evolve = np.empty([nrep, len(tvec)])

    mean =  freqpars[0]   #controls the frequency of the revivals
    delta =  freqpars[1]  #increases the appearance of secondary peaks
    sigma =   freqpars[2] # controls the T2 decay, as well as how "clean" the revivals will be 
    magoffset = Bpars[0] # controls the visibility of initial peaks (<0.5), but also the final collapse value, influences the width of the peaks
    sigmaO = Bpars[1] # controls the T2 decay, as well as how "clean" the revivals will be 
        
    for repeats in range(nrep): #needed to reduce fluctuations due to distribution sampling
    
#         freqsa = np.repeat(mean+delta, simsize) 
        freqsa = np.random.normal(loc=mean+delta, scale=sigma, size=Nqubit)   
        freqsb = np.random.normal(loc=mean, scale=sigma, size=Nqubit)    
        magoffsets = np.random.normal(loc=magoffset, scale=sigmaO, size=Nqubit)

        for idt in range(len(tvec)):

            shifted_t = tvec[idt]-offset
            S = [1 - magoffsets[i]*((np.sin(freqsa[i]*shifted_t/2))**2)*((np.sin(freqsb[i]*shifted_t/2))**2)   for i in range(len(freqsa))]

            evolve[repeats, idt] = (np.prod(S)+1)/2

    evolve = np.mean(evolve, axis=0)
        
    return evolve