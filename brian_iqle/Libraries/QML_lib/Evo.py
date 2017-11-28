import qinfer as qi
import numpy as np
import scipy as sp

try: 
    import hamiltonian_exponentiation as h
    ham_exp_installed = True
except:
    ham_exp_installed = False
     
## Generic states and Pauli matrices ##########################################################

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

def paulilist():
    return [sigmax(), sigmay(), sigmaz()]
    
    
## Functions for evolution ##########################################################

def getH(_pars, _ops):
    #return np.sum(pars*ops, axis=0)
    return (np.tensordot(_pars, _ops, axes=1))[0]

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



def pr0fromScipyNC(tvec, modpar, exppar, oplist, probestate, Hp = None, trotterize=True, use_exp_ham=ham_exp_installed):
    """Generic version to be adopted in case oplist includes non-commutative operators"""
#    use_exp_ham = True
#    print("pr0fromScipyNC using exp_ham: ", use_exp_ham)
    print_exp_ham=True
    evo = np.empty([len(modpar), len(tvec)])
    #dimension check
    if len(np.shape(oplist)) != 3:
        raise IndexError('OperatorList has the wrong shape')
    
    if not(all(np.shape(modpar)[1] == np.repeat(np.shape(oplist)[0], len(modpar)))):
        raise AttributeError('Shapes of OperatorList and Parameters do not match')

    #evolution for the system experimental parameters
    Hm = getH(exppar, oplist)
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
            
            if trotterize is False:
                if use_exp_ham:
                  backstate = np.dot(h.exp_ham(Hm, tvec[idt], plus_or_minus=1.0, print_method=print_exp_ham), probestate)
                  evostate = np.dot(h.exp_ham(Hp, tvec[idt], plus_or_minus=1.0,print_method=print_exp_ham), backstate)
                else: 
                  backstate = np.dot(sp.linalg.expm((1j)*tvec[idt]*Hm), probestate)
                  evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*Hp), backstate)
                #print('Evostate: ', evostate)
                
            else:
                # print('trotter')
                if use_exp_ham:
                  evostate = np.dot(h.exp_ham(Hp-Hm, tvec[idt], plus_or_minus=1.0,print_method=print_exp_ham), probestate)
                else: 
                  evostate = np.dot(sp.linalg.expm(-(1j)*tvec[idt]*(Hp-Hm)), probestate)

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
