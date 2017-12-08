from __future__ import print_function # so print doesn't show brackets
import qinfer as qi
import numpy as np
import scipy as sp
#import qutip as qt
import sys as sys
import os as os

try: 
    import hamiltonian_exponentiation as h
    # TODO set to true after testing
    ham_exp_installed = True
    
except:
    ham_exp_installed = False
    
    
import IOfuncts as mIO 

sys.path.append((os.path.join("..")))
import SETTINGS

## Generic states and Pauli matrices ##########################################################

global debug_print
debug_print = False

def plus():
    return np.array([1, 1])/np.sqrt(2)

def minus():
    return np.array([1, -1])/np.sqrt(2)
    
def zero():
    return np.array([1, 0])
    
def one():
    return np.array([0, 1])
    
def plusI():
    return np.array([1, 1j])/np.sqrt(2)
    
def minusI():
    return np.array([1, -1j])/np.sqrt(2)

def sigmaz():
    return np.array([[1+0.j, 0+0.j], [0+0.j, -1+0.j]])

def sigmax():
    return np.array([[0+0.j, 1+0.j], [1+0.j, 0+0.j]])

def sigmay():
    return np.array([[0+0.j, 0-1.j], [0+1.j, 0+0.j]])

def identity():
    return np.array([[1+0.j, 0+0.j], [0+0.j, 1+0.j]])


def paulilst():
    return [sigmax(), sigmay(), sigmaz()]
    
    """!!!!!!!!!!! Defined twice in different ways among me & Brian, argh XD
    we might want to unify at some point here - AAG"""
def paulilist():
    return [sigmax(), sigmay(), sigmaz()]
    
    
## Functions for evolution ##########################################################

def getH(_pars, _ops):
    #return np.sum(pars*ops, axis=0)
    return (np.tensordot(_pars, _ops, axes=1))[0]

# TODO: I changed this to give back total array, not just 0th element -- is that a problem? -Brian
#    return (np.tensordot(_pars, _ops, axes=1))

def pr0fromScipy(tvec, dw, oplist, probestate):
    """Version to be adopted only if operators in oplist commute"""
    
    evo = np.empty([len(dw), len(tvec)])
    
    #dimension check
    if len(np.shape(oplist)) != 3:
        raise IndexError('OperatorList has the wrong shape')
    
    if not(all(np.shape(dw)[1] == np.repeat(np.shape(oplist)[0], len(dw)))):
        raise AttributeError('Shapes of OperatorList and Parameters do not match')

    for evoidx in range(len(dw)):
        H = getH(dw[evoidx, np.newaxis], oplist)
        for idt in range(len(tvec)):
            unitary = sp.linalg.expm(-(1j)*tvec[idt]*H)
            evostate = np.dot(unitary, probestate)
            #print('Evostate: ', evostate)
            evo[evoidx][idt] = np.abs(np.dot(probestate, evostate.conj().T)) ** 2
    
    return evo



def pr0fromScipyNC(tvec, modpar, exppar, oplist, probestate, Hp = None, trotterize=True, IQLE=True, use_exp_ham=ham_exp_installed):
    """Generic version to be adopted in case oplist includes non-commutative operators"""
    
    print_exp_ham=False
    #    print("pr0fromScipyNC using exp_ham: ", use_exp_ham)
    
    evo = np.empty([len(modpar), len(tvec)])

    
    #dimension check
    if len(np.shape(oplist)) != 3:
        raise IndexError('OperatorList has the wrong shape')
    
    if not(all(np.shape(modpar)[1] == np.repeat(np.shape(oplist)[0], len(modpar)))):
        print("modpar has shape", np.shape(modpar))
        print(modpar)
        print("ops has shape ", np.shape(oplist))
        print(oplist)
        raise AttributeError('Shapes of OperatorList and Parameters do not match')

    #evolution for the system experimental parameters
    if IQLE is True:
        Hm = getH(exppar, oplist)
#        Hm = np.tensordot(exppar, oplist, axes=1) #TODO -- checking if this can fix an error -Brian
        if debug_print: print("in pr0fromScipyNC, within IQLE TRUE Hm = ")
        if debug_print: print(Hm) 
      
    else:
        Hm = None
    #print(Hm)
   
    if Hp is None or len(modpar)>1:
        trueEvo = False		# call the system with the tested Hamiltonian (or the simulator Hamiltonian for particles)
    else:
        trueEvo = True		# call the system with the "true" Hamiltonian
    
    for evoidx in range(len(modpar)):    
        #evolution for the system and particles in the simulator, assuming trueHam = simHam
        if not trueEvo:
            Hp = getH(modpar[evoidx, np.newaxis], oplist)
            #Hp = np.tensordot(modpar[evoidx, np.newaxis], oplist, axes=1)
        
        for idt in range(len(tvec)):
            
            # QLE evolution
            if Hm is None:
                if use_exp_ham:
                #TODO should exp_ham be used here?
                    evostate = np.dot(h.exp_ham(Hp, tvec[idt], plus_or_minus=1.0,print_method=print_exp_ham), backstate)
                else:
                    evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*Hp), probestate)
            
            # IQLE evolution
            else:
                if trotterize is False:
                    if use_exp_ham:
                        backstate = np.dot(h.exp_ham(Hm, tvec[idt], plus_or_minus=1.0, print_method=print_exp_ham), probestate)
                        evostate = np.dot(h.exp_ham(Hp, tvec[idt], plus_or_minus=1.0,print_method=print_exp_ham), backstate)
                    else:
                        backstate = np.dot(sp.linalg.expm((1j)*tvec[idt]*Hm), probestate)
                        evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*Hp), backstate)
                else:
                    # 0th order Trotterization for the evolution
                    if use_exp_ham:
                        #print("Using Exp ham custom")
                        HpMinusHm = Hp - Hm
    
                        if debug_print: print("Before exponentiation, Hm & Hp : ")
                        if debug_print: print(Hm)
                        if debug_print: print(Hp)
                        if debug_print: print("Giving mtx of shape ", np.shape(HpMinusHm))
                        if debug_print: print(Hp-Hm)
                        if debug_print: print("t= ", tvec[idt])

                        # TODO HpMinusHm has np.shape (1,2,2) --  should be (2,2) for passing to exp_ham
                        unitary = h.exp_ham(HpMinusHm, tvec[idt], plus_or_minus=1.0,print_method=print_exp_ham)
                        evostate = np.dot(unitary, probestate)
                    else:
                        evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*(Hp-Hm)), probestate)
                #if debug_print: print('Evostate: ', evostate)
        
            evo[evoidx][idt] = np.abs(np.dot(evostate.conj(), probestate.T)) ** 2         
    
    return evo
    
    
    
    
    
def pr0fromScipyNCpp(tvec, modpar, exppar, oplist, probestate, Hp = None, trotterize=True):
    """Parallelized-ready version of pr0fromScipyNC"""
    if len(modpar.shape) == 1:
            modpar = np.array([modpar])
            
    evo = np.empty([len(modpar), len(tvec)])
    
    #dimension check
    if len(np.shape(oplist)) != 3:
        raise IndexError('OperatorList has the wrong shape')
    
    if not(all(np.shape(modpar)[1] == np.repeat(np.shape(oplist)[0], len(modpar)))):
        raise AttributeError('Shapes of OperatorList and Parameters do not match')

    #evolution for the system experimental parameters
    Hm = getH(exppar, oplist)
    #if debug_print: print(Hm)
   
    if Hp is None or len(modpar)>1:
        trueEvo = False		# call the system with the tested Hamiltonian (or the simulator Hamiltonian for particles)
        #print("Calling the false Hamiltonian")
    else:
        trueEvo = True		# call the system with the "true" Hamiltonian
        #print("Calling the true Hamiltonian")
    
    for evoidx in range(len(modpar)):
        #evolution for the system and particles in the simulator, assuming trueHam = simHam
        if not trueEvo:
            Hp = getH(modpar[evoidx, np.newaxis], oplist)
        #if debug_print: print(Hp)
        
        for idt in range(len(tvec)):
            
            if trotterize is False:
                backstate = np.dot(sp.linalg.expm((1j)*tvec[idt]*Hm), probestate)
                evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*Hp), backstate)
                #print('Evostate: ', evostate)
                
            else:
                # print('trotter')
                evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*(Hp-Hm)), probestate)
        
            evo[evoidx][idt] = np.abs(np.dot(evostate.conj(), probestate.T)) ** 2         
    
    if len(modpar.shape) == 1:
        return (evo[0][0])
    else:
        return evo
        
        

def pr0fromHahnPeak(tvec, modpar, exppar, oplist, probestate, Hp = None, trotterize=True, IQLE=True):
    """Version dedicated to obtain likelihoods from Hahn-echo peak (i.e. tau != tau') experiments"""
    
    evo = np.empty([len(modpar), len(tvec)])
    
    Hahn_angle = np.pi/2
    H_Hahn = np.kron(Hahn_angle*sigmaz(),np.eye(2))
    
    #dimension check
    if len(np.shape(oplist)) != 3:
        raise IndexError('OperatorList has the wrong shape')
    
    if not(all(np.shape(modpar)[1] == np.repeat(np.shape(oplist)[0], len(modpar)))):
        raise AttributeError('Shapes of OperatorList and Parameters do not match')

    #evolution for the system experimental parameters
    if IQLE is True:
        Hm = getH(exppar, oplist)
    else:
        Hm = None
    #if debug_print: print(Hm)
   
    if Hp is None or len(modpar)>1:
        trueEvo = False		# call the system with the tested Hamiltonian (or the simulator Hamiltonian for particles)
        #print("Calling the false Hamiltonian")
    else:
        trueEvo = True		# call the system with the "true" Hamiltonian
        #print("Calling the true Hamiltonian")
    
    for evoidx in range(len(modpar)):    
        #evolution for the system and particles in the simulator, assuming trueHam = simHam
        if not trueEvo:
            Hp = getH(modpar[evoidx, np.newaxis], oplist)
        #if debug_print: print(Hp)
        
        
        for idt in range(len(tvec)):
            
            # QLE evolution
            if Hm is None:

                evostate = np.dot(sp.linalg.expm(-(1j)*1*H_Hahn), probestate)    # Hahn-echo operation

                evostate = np.dot(sp.linalg.expm(-(1j)*2*tvec[idt]*Hp), evostate)
     
               
            
            # IQLE evolution
            else:
                print("IQLE not implemented")                
                # if trotterize is False:
                    # backstate = np.dot(sp.linalg.expm((1j)*tvec[idt]*Hm), probestate)
                    # evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*Hp), backstate)
                # else:
                    # # 0th order Trotterization for the evolution
                    # evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*(Hp-Hm)), probestate)
                # # print('Evostate: ', evostate)
                
            
            
            ### Added here the conversion to QuTip object
            qt_evostate = qt.Qobj(evostate) 
            qt_evostate.dims = [[2,2],[1,1]]

            
            ## Modified here the expectation as 
            evo[evoidx][idt] = 1-(qt.expect(qt_evostate.ptrace(0), qt.Qobj(plus()) ))**2

                     
    return evo
    
    
    
def EXPOFFpr0fromHahnPeak(tvec, modpar, exppar, oplist, probestate, Hp = None, trotterize=True, IQLE=True):
    """Version dedicated to obtain likelihoods from Hahn-echo peak (i.e. tau != tau') experiments
    Reads data from vector of OFFLINE taken data
    """
    
    evo = np.empty([len(modpar), len(tvec)])
    
    Hahn_angle = np.pi/2
    H_Hahn = np.kron(Hahn_angle*sigmaz(),np.eye(2))
    
    #dimension check
    if len(np.shape(oplist)) != 3:
        raise IndexError('OperatorList has the wrong shape')
    
    if not(all(np.shape(modpar)[1] == np.repeat(np.shape(oplist)[0], len(modpar)))):
        raise AttributeError('Shapes of OperatorList and Parameters do not match')

    #evolution for the system experimental parameters
    if IQLE is True:
        Hm = getH(exppar, oplist)
    else:
        Hm = None
    #print(Hm)
   
    if Hp is None or len(modpar)>1:
        trueEvo = False		# call the system with the tested Hamiltonian (or the simulator Hamiltonian for particles)
        #print("Calling the false Hamiltonian")
    else:
        trueEvo = True		# call the system with the "true" Hamiltonian
        #print("Calling the true Hamiltonian")
    
    for evoidx in range(len(modpar)):    
        #evolution for the system and particles in the simulator, assuming trueHam = simHam
        if not trueEvo:
            Hp = getH(modpar[evoidx, np.newaxis], oplist)
            #print(Hp)
        
        
            for idt in range(len(tvec)):
                
                # QLE evolution
                if Hm is None:

                    evostate = np.dot(sp.linalg.expm(-(1j)*1*H_Hahn), probestate)    # Hahn-echo operation
                    evostate = np.dot(sp.linalg.expm(-(1j)*2*tvec[idt]*Hp), evostate)
  
                # IQLE evolution
                else:
                    print("IQLE not implemented")                

            
                ### Added here the conversion to QuTip object
                qt_evostate = qt.Qobj(evostate) 
                qt_evostate.dims = [[2,2],[1,1]]

                
                ## Modified here the expectation as 
                evo[evoidx][idt] = 1-(qt.expect(qt_evostate.ptrace(0), qt.Qobj(plus()) ))**2

        else: # call experimental data
            
            for idt in range(len(tvec)):
                #print('The t param is defined as: ' + repr(tvec[idt]))
                evo[evoidx][idt] = (mIO.EXPfromVector(SETTINGS.mydata, tvec[idt]))[1]
                
                # if len(evo) < 2:
                    # print('found P0: ' + repr(evo))
                     
    return evo
    
    

def pr0fromHahnAnaSignal(tvec, modpar, exppar, Nqubit, IQLE=False):
    """Version dedicated to obtain likelihoods from Hahn-echo signal (i.e. tau = tau') experiments
    adopting the likelihood provided by the theoretical RWA-approximated model for single qubit:
    S = $[1 - B_envelope*((\sin(\omega_0[i]*t/2))^2)*((\sin(\omega_1[i]*t/2))^2)]$
    """
    nrep = 5
    evo = np.empty([len(modpar), len(tvec)])
   
    if len(modpar)>1:
        trueEvo = False		# call the system with the simulator Hamiltonian for particles)
        # print("Calling the false Hamiltonian")
    else:
        trueEvo = True		# call the system with the "true" Hamiltonian
        # print("Calling the true Hamiltonian")
    
    for evoidx in range(len(modpar)):    
        
        modpars = modpar[evoidx] #np.newaxis
        
        #evolution for the system and particles in the simulator, assuming trueHam = simHam
        
        for idt in range(len(tvec)):
            
            # QLE only evolution
            if IQLE is False:

                mean = modpars[0]   #controls the frequency of the revivals
                delta = modpars[1]  #increases the appearance of secondary peaks
                sigma =  max(modpars[2], 10**-12)   # controls the T2 decay, as well as how "clean" the revivals will be 
                magoffset = 0.45 # modpars[2]  # controls the visibility of initial peaks (<0.5), but also the final collapse value, influences the width of the peaks
                sigmaO = 0.015
                
                evolve = np.empty(nrep)
                for repeats in range(nrep): #average to mitigate statistical fluctuations

                    #Generate an appropriate sample of omega_0 and omega_1 frequencies
                    #for the nuclear ensemble
                    freqs1 = np.random.normal(loc=mean+delta, scale=sigma, size=Nqubit)   
                    freqs2 = np.random.normal(loc=mean, scale=sigma, size=Nqubit)  
                    
                    magoffsets = np.random.normal(loc=magoffset, scale=sigmaO, size=Nqubit)
                    # magoffsets = np.repeat(magoffset, Nqubit)                     
                    
                    ## TODO change here for simulated data
                    # shift t in order to match experimental data
                    shifted_t = tvec[idt] # -SETTINGS.signal_offset
                    
                    # Compute the likelihood according to the theoretical model in Childress07
                    S = [1 - magoffsets[i]*((np.sin(freqs1[i]*shifted_t/2))**2)*((np.sin(freqs2[i]*shifted_t/2))**2)   for i in range(len(freqs1))]
                   
                    evolve[repeats] = (np.prod(S)+1)/2
                
                evo[evoidx][idt] = np.mean(evolve, axis=0)            
            
            # IQLE evolution 
            ## TODO might be implemented, yet it does not look trivial because of the statistical ensemble appearing
            else:               
                raise(ValueError('IQLE not implemented'))
        
                     
    return evo
    
    
    
    
    
    
def EXPOFFpr0fromHahnSignal(tvec, modpar, exppar, Nqubit, IQLE=False):
    """Version dedicated to obtain likelihoods from Hahn-echo peak (i.e. tau != tau') experiments
    Reads data from vector of OFFLINE taken data
    """
    nrep = 1
    evo = np.empty([len(modpar), len(tvec)])
    
    if len(modpar)>1:
        trueEvo = False		# call the system with the simulator Hamiltonian for particles)
        # print("Calling the false Hamiltonian")
    else:
        trueEvo = True		# call the system with the "true" Hamiltonian
        # print("Calling the true Hamiltonian")
    
    
    #evolution for the system and particles in the simulator, assuming trueHam = simHam
    if not trueEvo:
    
        for evoidx in range(len(modpar)):    
  
            modpars = modpar[evoidx] #np.newaxis
            
            #evolution for the system and particles in the simulator, assuming trueHam = simHam
            
            for idt in range(len(tvec)):
                
                # QLE only evolution
                if IQLE is False:

                    mean = modpars[0]   #controls the frequency of the revivals
                    delta = modpars[1]  #increases the appearance of secondary peaks
                    sigma =  max(modpars[2], 10**-12)   # controls the T2 decay, as well as how "clean" the revivals will be 
                    magoffset = 0.45 # modpars[2]  # controls the visibility of initial peaks (<0.5), but also the final collapse value, influences the width of the peaks
                    sigmaO = 0.015
                    
                    evolve = np.empty(nrep)
                    for repeats in range(nrep): #average to mitigate statistical fluctuations

                        #Generate an appropriate sample of omega_0 and omega_1 frequencies
                        #for the nuclear ensemble
                        freqs1 = np.random.normal(loc=mean+delta, scale=sigma, size=Nqubit)   
                        freqs2 = np.random.normal(loc=mean, scale=sigma, size=Nqubit)  
                        
                        magoffsets = np.random.normal(loc=magoffset, scale=sigmaO, size=Nqubit)
                        # magoffsets = np.repeat(magoffset, Nqubit)                     
                        
                        ## TODO change here for simulated data
                        # shift t in order to match experimental data
                        shifted_t = tvec[idt] # -SETTINGS.signal_offset
                        
                        # Compute the likelihood according to the theoretical model in Childress07
                        S = [1 - magoffsets[i]*((np.sin(freqs1[i]*shifted_t/2))**2)*((np.sin(freqs2[i]*shifted_t/2))**2)   for i in range(len(freqs1))]
                       
                        evolve[repeats] = (np.prod(S)+1)/2
                    
                    evo[evoidx][idt] = np.mean(evolve, axis=0)            
                
                # IQLE evolution 
                ## TODO might be implemented, yet it does not look trivial because of the statistical ensemble appearing
                else:               
                    raise(ValueError('IQLE not implemented'))
                    
    
    else: #call experimental data
    
        for evoidx in range(len(modpar)):    
        
            for idt in range(len(tvec)):
                # print('The t param is defined as: ' + repr(tvec[idt]))
                evo[evoidx][idt] = (mIO.EXPfromVector(SETTINGS.sigdata, tvec[idt]))[1]
                
                # if len(evo) < 2:
                    # print('found P0: ' + repr(evo))
                     
    return evo
        
        
        
def pr0fromHahn(tvec, modpar, exppar, oplist, probestate, Hp = None, trotterize=True, IQLE=True):
    """Version dedicated to obtain likelihoods from Hahn-echo experiments"""
    
    evo = np.empty([len(modpar), len(tvec)])
    
    Hahn_angle = np.pi/2
    H_Hahn = np.kron(Hahn_angle*sigmaz(),np.eye(2))
    
    #dimension check
    if len(np.shape(oplist)) != 3:
        raise IndexError('OperatorList has the wrong shape')
    
    if not(all(np.shape(modpar)[1] == np.repeat(np.shape(oplist)[0], len(modpar)))):
        raise AttributeError('Shapes of OperatorList and Parameters do not match')

    #evolution for the system experimental parameters
    if IQLE is True:
        Hm = getH(exppar, oplist)
    else:
        Hm = None
    #print(Hm)
   
    if Hp is None or len(modpar)>1:
        trueEvo = False		# call the system with the tested Hamiltonian (or the simulator Hamiltonian for particles)
        #print("Calling the false Hamiltonian")
    else:
        trueEvo = True		# call the system with the "true" Hamiltonian
        #print("Calling the true Hamiltonian")
    
    for evoidx in range(len(modpar)):    
        #evolution for the system and particles in the simulator, assuming trueHam = simHam
        if not trueEvo:
            Hp = getH(modpar[evoidx, np.newaxis], oplist)
        #print(Hp)
        
        
        for idt in range(len(tvec)):
            
            # QLE evolution
            if Hm is None:
            
                evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*Hp/2), probestate)

                evostate = np.dot(sp.linalg.expm(-(1j)*1*H_Hahn), evostate)    # Hahn-echo operation

                evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*Hp/2), evostate)
     
               
            
            # IQLE evolution
            else:
                print("IQLE not implemented")                
                # if trotterize is False:
                    # backstate = np.dot(sp.linalg.expm((1j)*tvec[idt]*Hm), probestate)
                    # evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*Hp), backstate)
                # else:
                    # # 0th order Trotterization for the evolution
                    # evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*(Hp-Hm)), probestate)
                # # print('Evostate: ', evostate)
                
            
            
            ### Added here the conversion to QuTip object
            qt_evostate = qt.Qobj(evostate) 
            qt_evostate.dims = [[2,2],[1,1]]

            
            ## Modified here the expectation as 
            evo[evoidx][idt] = qt.expect(qt_evostate.ptrace(0), qt.Qobj(minus()) )
            #evo[evoidx][idt] = np.abs(np.dot(evostate.conj(), probestate.T)) ** 2

        # else:
            
            # for idt in range(len(tvec)):
                # evo[evoidx][idt] = pr0EXPfromHahn(tvec[idt])
                     
    return evo
