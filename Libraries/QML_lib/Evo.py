from __future__ import print_function # so print doesn't show brackets
import qinfer as qi
import qutip as qt
import numpy as np
import scipy as sp
#import qutip as qt
import sys as sys
import os as os

use_linalg = False


try: 
    import hamiltonian_exponentiation as h
    # TODO set to true after testing
    ham_exp_installed = True
    
except:
    ham_exp_installed = False
    

if(use_linalg):
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




def get_pr0_array(t_list, sim_params, sim_ops, true_ham, trotterize=True):
    
    num_params = sim_params.shape[1]
    num_times = len(t_list)
    
    print("num parms : ", num_params)
    print("num times : ", num_times)
    
    print("get pr0; t_list =", t_list)
    #print("sim params ", sim_params)
    print("sim params has shape ", sim_params.shape)
    print("true_ham = ", true_ham)
#    print("true params :", true_params)
#    print("true ops : ", true_ops)

#    print("sim params :", sim_params)
#    print("sim ops : ", sim_ops)
    sim_ham = getH(sim_params, sim_ops)

#    true_ham = getH(true_params, true_ops)
    
#    print("Sim ham = ", sim_ham)
#    print("True ham = ", true_ham)
    output = np.empty([num_params, num_times])
#    evo = np.empty([len(modpar), len(tvec)])
    print("output has shape ", output.shape)
   # print("sim params has shape : ", np.shape(sim_params))
    print("len sim params : ", len(sim_params[0,:]))
    print("len t list : ", len(t_list))
    for evoId in range( len(sim_params[0,:])): ## todo not sure about length/arrays here
        for tId in range(len(t_list)):
            t = t_list[tId]
            
            print("evoId=", evoId, "\t tId = ", tId)
            output[evoId][tId] = overlap(ham_true = true_ham, ham_sim=sim_ham, t=t)
    print("Returning output : ", output)
    return output


def pr0fromScipyNC(tvec, modpar, exppar, oplist, probestate, Hp = None, trotterize=True, IQLE=True, use_exp_custom=ham_exp_installed):
    """Generic version to be adopted in case oplist includes non-commutative operators"""
    
    print_exp_ham=False
    """
    if use_exp_custom: 
        print("pr0fromScipyNC using exp_ham ")
    else: 
        print("pr0fromScipyNC using linalg")
    """
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
                if use_exp_custom:
                #TODO should exp_ham be used here?
#                    evostate = np.dot(h.exp_ham(Hp, tvec[idt], plus_or_minus=1.0,print_method=print_exp_ham), backstate)
                    evostate = np.dot(h.exp_ham(Hp, tvec[idt], plus_or_minus=1.0,print_method=print_exp_ham), probestate)
                else:
                    evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*Hp), probestate)
            
            # IQLE evolution
            else:
                if trotterize is False:
                    if use_exp_custom:
                        backstate = np.dot(h.exp_ham(Hm, tvec[idt], plus_or_minus=1.0, print_method=print_exp_ham), probestate)
                        evostate = np.dot(h.exp_ham(Hp, tvec[idt], plus_or_minus=1.0,print_method=print_exp_ham), backstate)
                    else:
                        backstate = np.dot(sp.linalg.expm((1j)*tvec[idt]*Hm), probestate)
                        evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*Hp), backstate)
                else:
                    # 0th order Trotterization for the evolution
                    if use_exp_custom:
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
    

# def pr0_partial_trace(tvec, modpar, exppar, oplist, probestate, Hp = None, trotterize=True, IQLE=True, use_exp_ham=ham_exp_installed):
    



    
    
    
    
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
                
            
            
            qt_evostate = qt.Qobj(evostate) 
            ### Added here the conversion to QuTip object
            qt_evostate.dims = [[2,2],[1,1]]

            
            ## Modified here the expectation as 
            evo[evoidx][idt] = qt.expect(qt_evostate.ptrace(0), qt.Qobj(minus()) )
            #evo[evoidx][idt] = np.abs(np.dot(evostate.conj(), probestate.T)) ** 2

        # else:
            
            # for idt in range(len(tvec)):
                # evo[evoidx][idt] = pr0EXPfromHahn(tvec[idt])
                     
    return evo
    
    
    
## Partial trace functionality

def expectation_value(ham, t, state=None, choose_random_probe=False):
    if choose_random_probe is True: 
        num_qubits = int(np.log2(np.shape(ham)[0]))
        state = random_probe(num_qubits)
    elif random_probe is False and state is None: 
        print ("expectation value function: you need to either pass a state or set choose_random_probe=True")
    u_psi = evolved_state(ham, t, state)
    probe_bra = state.conj().T
    psi_u_psi = np.dot(probe_bra, u_psi)
    return np.abs(psi_u_psi**2)

def evolved_state(ham, t, state):
    import hamiltonian_exponentiation as h
    unitary = h.exp_ham(ham, t)
    return np.dot(unitary, state)

def random_probe(num_qubits):
    dim = 2**num_qubits
    real = np.random.rand(1,dim)
    imaginary = np.random.rand(1,dim)
    complex_vectors = np.empty([1, dim])
    complex_vectors = real +1.j*imaginary

    vals_squared = [val**2 for val in complex_vectors]
    norm_factor = np.sqrt(np.sum(vals_squared))
    probe = complex_vectors/norm_factor
    return probe[0][:]

def trim_vector(state, final_num_qubits):
    return state[:2**final_num_qubits]

def qutip_evolved_state(ham, t, state):
    evolved = evolved_state(ham,t,state=state)
    return qt.Qobj(evolved)


def overlap(ham_true, ham_sim, t):
    print("overlap :")
    print("ham true :" , ham_true)
    print("ham sim :", ham_sim)
    print("t=", t)


    true_dim = int(np.log2(np.shape(ham_true)[0])) 
    sim_dim = int(np.log2(np.shape(ham_sim)[0]))

    min_dim = min(true_dim, sim_dim)
    max_dim = max(true_dim, sim_dim)
    to_keep = range(min_dim)

    probe = random_probe(max_dim)
    reduced_probe = trim_vector(probe, final_num_qubits=min_dim)
    
    
    if sim_dim > min_dim: 
        sim = qutip_evolved_state(ham_sim, t, probe)
        sim_reduced = sim.ptrace(to_keep)
    else:
        sim =  qutip_evolved_state(ham_sim, t, reduced_probe)
        sim_reduced = sim.ptrace(to_keep)
    
    if true_dim > min_dim: 
        true = qutip_evolved_state(ham_true, t, probe)
        true_reduced = true.ptrace(to_keep)
    else: 
        true = qutip_evolved_state(ham_true, t, reduced_probe)
        true_reduced = true.ptrace(to_keep)
        
    #return true_reduced, sim_reduced
    overlap = qt.expect(sim_reduced, true_reduced)
    print("overlap is ", overlap)
    return overlap
    
