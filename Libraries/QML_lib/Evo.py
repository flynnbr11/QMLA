from __future__ import print_function # so print doesn't show brackets
import qinfer as qi
#import qutip as qt
import numpy as np
import scipy as sp
import inspect

#import qutip as qt
import sys as sys
import os as os
import IOfuncts as mIO 
from MemoryTest import print_loc
sys.path.append((os.path.join("..")))
#import SETTINGS


global_print_loc = False 
use_linalg = False
use_sparse = False 
global print_pr0
print_pr0 = False
 
 
try: 
    import hamiltonian_exponentiation as h
    # TODO set to true after testing
    ham_exp_installed = True
     
except:
    ham_exp_installed = False
     
 
if(use_linalg):
    ham_exp_installed = False
     

#print("Using exp ham custom : ", ham_exp_installed)


     
 
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

def identity():
    return np.array([[1+0.j, 0+0.j], [0+0.j, 1+0.j]])
 
 
def sigmaz():
    return np.array([[1+0.j, 0+0.j], [0+0.j, -1+0.j]])
 
def sigmax():
    return np.array([[0+0.j, 1+0.j], [1+0.j, 0+0.j]])
 
def sigmay():
    return np.array([[0+0.j, 0-1.j], [0+1.j, 0+0.j]])
 
 
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


def time_seconds():
    import datetime
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    return time


def log_print(to_print_list, log_file, log_identifier=None):
    identifier = str(str(time_seconds()) +" [Evo ("+str(log_identifier)+")]")
    if type(to_print_list)!=list:
        to_print_list = list(to_print_list)

    print_strings = [str(s) for s in to_print_list]
    to_print = " ".join(print_strings)
    with open(log_file, 'a') as write_log_file:
        print(identifier, str(to_print), file=write_log_file)
 
 
def anaytical_pr0(t_list, modelparams, oplist, probe):
    print_loc(global_print_loc)
    num_particles = len(modelparams)

    #if num_particles==1: print("True probe \n", probe)
    #else: print("Simulated probe \n", probe)
    num_times = len(t_list)
    output = np.empty([num_particles, num_times])

    for evoId in range(num_particles): ## todo not sure about length/arrays here
#        ham = np.tensordot(modelparams[evoId], oplist, axes=1)
        for tId in range(len(t_list)):
            t = t_list[tId]
            output[evoId][tId] = (np.cos(modelparams[evoId] * t))**2
    return output

 
def get_pr0_array_qle(t_list, modelparams, oplist, probe, use_exp_custom=True,exp_comparison_tol=None, enable_sparse=True, ham_list=None, log_file='QMDLog.log', log_identifier=None):
    
    print_loc(global_print_loc)
    num_particles = len(modelparams)

    num_times = len(t_list)
    output = np.empty([num_particles, num_times])
    for evoId in range(num_particles): ## todo not sure about length/arrays here
        for tId in range(len(t_list)):
            ham = np.tensordot(modelparams[evoId], oplist, axes=1)
            t = t_list[tId]
            print_loc(global_print_loc)
            try:
                output[evoId][tId] = expectation_value(ham=ham, t=t, state=probe, use_exp_custom=use_exp_custom, compare_exp_fncs_tol=exp_comparison_tol, enable_sparse=enable_sparse, log_file=log_file, log_identifier=log_identifier)
            except NameError:
                log_print(["Error raised; unphysical expecation value."], log_file, log_identifier)
                sys.exit()
                log_print(["Inputs to expectation value function. \n\t ham=", ham, "\n\t t=", t, "\n\t state=",probe, "\n\t use_exp_custom=", use_exp_custom, "\n\t exp_comparison_tol=", exp_comparison_tol, "\n\t enable_sparse", enable_sparse, "\n\t log_file=", log_file, "\n\t log_id=", log_identifier], log_file, log_identifier)
            
                
            if output[evoId][tId] < 0:
                log_print(["[QLE] Negative probability : \t \t probability = ", output[evoId][tId]], log_file, log_identifier)
            elif output[evoId][tId] > 1.001: ## todo some times getting p=1.0 show up
                log_print(["[QLE] Probability > 1: \t \t probability = ", output[evoId][tId]], log_file, log_identifier) 
            #print("(i,j) = (", evoId, tId,") \t val: ", output[evoId][tId])
    print_loc(global_print_loc) 
    return output
 

def get_pr0_array_iqle(t_list, modelparams, oplist, ham_minus, probe, use_exp_custom=True, enable_sparse=True, exp_comparison_tol=None, trotterize=True, ham_list = None, log_file='QMDLog.log',log_identifier=None):
    print_loc(global_print_loc)
    num_particles = len(modelparams)
    num_times = len(t_list)
     
 
    output = np.empty([num_particles, num_times])
 
    if print_pr0: print("output has shape ", output.shape)
 
    for evoId in range( output.shape[0]): ## todo not sure about length/arrays here
        ham = np.tensordot(modelparams[evoId], oplist, axes=1)
        for tId in range(len(t_list)):
            t = t_list[tId]
             
            output[evoId][tId] = iqle_evolve(ham = ham, ham_minus = ham_minus, t=t, probe=probe, use_exp_custom=use_exp_custom, compare_exp_fncs_tol=exp_comparison_tol,  enable_sparse=enable_sparse, log_file=log_file, log_identifier=log_identifier)
            print_loc(global_print_loc)
            if output[evoId][tId] < 0:
                log_print(["negative probability : \t \t probability = ", output[evoId][tId]], log_file, log_identifier)
            elif output[evoId][tId] > 1.000000000000001:
                log_print(["[IQLE] Probability > 1: \t \t probability = ", output[evoId][tId]], log_file, log_identifier) 
            #print("(i,j) = (", evoId, tId,") \t val: ", output[evoId][tId])
    print_loc(global_print_loc)
    return output

     
     
     
## Partial trace functionality
 
def expectation_value(ham, t, state=None, choose_random_probe=False, use_exp_custom=True, enable_sparse=True, print_exp_details=False, exp_fnc_cutoff=20, compare_exp_fncs_tol=None, log_file='QMDLog.log', log_identifier=None):
# todo: list of probes, maybe 5 is enough? test with different values
    print_loc(global_print_loc)
    print_loc(global_print_loc)
    if choose_random_probe is True: 
        num_qubits = int(np.log2(np.shape(ham)[0]))
        state = random_probe(num_qubits)
    elif random_probe is False and state is None: 
        log_print(["expectation value function: you need to either pass a state or set choose_random_probe=True"], log_file=log_file, log_identifier=log_identifier)
    print_loc(global_print_loc)
    
    if compare_exp_fncs_tol is not None:
        u_psi_linalg = evolved_state(ham, t, state, use_exp_custom=False, print_exp_details=print_exp_details, exp_fnc_cutoff=exp_fnc_cutoff)
        u_psi_exp_custom = evolved_state(ham, t, state, use_exp_custom=True, print_exp_details=print_exp_details, exp_fnc_cutoff=exp_fnc_cutoff)
        
        diff = np.max(np.abs(u_psi_linalg-u_psi_exp_custom))
        if np.allclose(u_psi_linalg, u_psi_exp_custom, atol=compare_exp_fncs_tol) == False:
            log_print(["Linalg/ExpHam give different evolved state by", diff], log_file=log_file, log_identifier=log_identifier)
            u_psi = u_psi_linalg
        else:
            u_psi = u_psi_exp_custom
            
    else:
        
        if use_exp_custom and ham_exp_installed:
          try:
              u_psi = evolved_state(ham, t, state, use_exp_custom=True, print_exp_details=print_exp_details, exp_fnc_cutoff=exp_fnc_cutoff)
          except ValueError:
              print("Value error when exponentiating Hamiltonian. Ham:\n", ham)
              print("Probe: ", state)
        else:
          u_psi = evolved_state(ham, t, state, use_exp_custom=False, print_exp_details=print_exp_details, exp_fnc_cutoff=exp_fnc_cutoff)
    
    
    print_loc(global_print_loc)
    probe_bra = state.conj().T
    
    
    psi_u_psi = np.dot(probe_bra, u_psi)
    
    print_loc(global_print_loc) 
    expec_value = np.abs(psi_u_psi)**2 ## TODO MAKE 100% sure about this!!
    
    expec_value_limit=1.1
#    expec_value_limit=0.1000
    
    if expec_value > expec_value_limit:
        log_print(["expectation value function has value ", np.abs(psi_u_psi**2)], log_file=log_file, log_identifier=log_identifier)
        log_print(["t=", t, "\nham = \n ", ham, "\nprobe : \n", state, "\nprobe normalisation:", np.linalg.norm(state), "\nU|p>:", u_psi, "\nnormalisation of U|p>:", np.linalg.norm(u_psi), "\n<p|U|p>:", psi_u_psi, "\nExpec val:", expec_value], log_file=log_file, log_identifier=log_identifier)
        log_print(["Recalculating expectation value using linalg."], log_file=log_file, log_identifier=log_identifier)
        u_psi = evolved_state(ham, t, state, use_exp_custom=False, log_file=log_file, log_identifier=log_identifier)
        psi_u_psi = np.dot(probe_bra, u_psi)
        expec_value = np.abs(psi_u_psi)**2 ## TODO MAKE 100% sure about this!!
    
      
    if expec_value > expec_value_limit:
        log_print(["Terminating due to expec value:", expec_value], log_file, log_identifier)
        log_print(["Testing evolved state fnc:"], log_file, log_identifier)
        expec_value = evolved_state(ham, t, state, use_exp_custom=True, print_exp_details=True, log_file=log_file, log_identifier=log_identifier)
        raise NameError('UnphysicalExpectationValue') 
        
    print_loc(global_print_loc)
    print_expec_value_intermediate = False
    if print_expec_value_intermediate:
      log_print(["Bra : \n", probe_bra], log_file=log_file, log_identifier=log_identifier)
      log_print(["u_psi\n", u_psi], log_file=log_file, log_identifier=log_identifier)
      log_print(["psi_psi:\n", np.dot(probe_bra, state)], log_file=log_file, log_identifier=log_identifier)
      log_print(["t=", t], log_file=log_file, log_identifier=log_identifier)
      log_print(["Ham : \n", ham], log_file=log_file, log_identifier=log_identifier)
      log_print(["probe : \n", state], log_file=log_file, log_identifier=log_identifier)
      log_print(["u_psi: \n", u_psi], log_file=log_file, log_identifier=log_identifier)
      log_print(["psi_u_psi: \n", psi_u_psi], log_file=log_file, log_identifier=log_identifier)
      log_print(["Expectation value : ", expec_value], log_file=log_file, log_identifier=log_identifier)
    return expec_value
 
def evolved_state(ham, t, state, use_exp_custom=True, enable_sparse=True, print_exp_details=False, exp_fnc_cutoff=10, log_file=None, log_identifier=None):
    #import hamiltonian_exponentiation as h
    from scipy import linalg
    print_loc(global_print_loc)
  
    if t>1e6: ## Try limiting times to use to 1 million
        t=1e6 # TODO PUT BACK IN. testing high t to find bug. 

    #print("Enable sparse : ", enable_sparse)    
    if use_exp_custom and ham_exp_installed:
        if log_file is not None:
            log_print(["Using custom expm. Exponentiating\nt=",t, "\nHam=\n", ham], log_file, log_identifier)
        unitary = h.exp_ham(ham, t, enable_sparse_functionality=enable_sparse, print_method=print_exp_details, scalar_cutoff=t+1)
    else:
      # print("Note: using linalg for exponentiating Hamiltonian.")
        if log_file is not None:
            iht = (-1j*ham*t)
            log_print(["Using linalg.expm. Exponentiating\nt=",t, "\nHam=\n", ham, "\n-iHt=\n", iht, "\nMtx elements type:", type(iht[0][0]), "\nMtx type:", type(iht)], log_file, log_identifier)
        unitary = linalg.expm(-1j*ham*t)
        
        if log_file is not None:
            log_print(["linalg.expm gives \nU=\n",unitary], log_file, log_identifier)
    
    print_loc(global_print_loc)
    ev_state = np.dot(unitary, state)
    if log_file is not None:
        log_print(["evolved state fnc. Method details printed in worker log. \nt=",t, "\nHam=\n", ham, "\nprobe=", state, "\nU=\n", unitary, "\nev_state=", ev_state], log_file, log_identifier)
    del unitary # to save space
    print_loc(global_print_loc)
    return ev_state


 
def random_probe(num_qubits):
    dim = 2**num_qubits
    real = []
    imaginary = []
    complex_vectors = []
    for i in range(dim):
        real.append(np.random.uniform(low=-1, high=1))
        imaginary.append(np.random.uniform(low=-1, high=1))
        complex_vectors.append(real[i] + 1j*imaginary[i])

    a=np.array(complex_vectors)
    norm_factor = np.linalg.norm(a)
    probe = complex_vectors/norm_factor
    if np.isclose(1.0, np.linalg.norm(probe), atol=1e-14) is False:
        print("Probe not normalised. Norm factor=", np.linalg.norm(probe)-1)
        return random_probe(num_qubits)

    return probe
def one_zeros_probe(num_qubits):
    dim = 2**num_qubits
    real = np.zeros(dim)
    imaginary = np.zeros(dim)
    real[0] = 1.0
    complex_vectors = np.empty([dim])
    complex_vectors = real +1.j*imaginary
    probe = complex_vectors/1.0
    return probe
 
 
  
def outer_product(state, as_qutip_object=False):
    dim = int((state.shape[0]))
    if as_qutip_object:
        return qt.Qobj(np.kron(state.conj(), state).reshape(dim, dim))
    else: 
        return np.kron(state.conj(), state).reshape(dim, dim) 
  
 
#import qutip as qt
 
def iqle_evolve(ham, ham_minus, t, probe, trotterize=True, use_exp_custom=True, enable_sparse=True, log_file='QMDLog.log', log_identifier=None):
    print_loc(global_print_loc)
    ham_dim = int(np.log2(np.shape(ham)[0])) 
    ham_minus_dim = int(np.log2(np.shape(ham_minus)[0]))
 
 
    if trotterize == True: 
        if ham_dim == ham_minus_dim: 
            H = ham_minus - ham ##reversed because exp_ham function calculated e^{-iHt}
            print_loc(global_print_loc)
            expec_value = expectation_value(H, t, state=probe, use_exp_custom=use_exp_custom, enable_sparse=enable_sparse)
            #print("expectation value : ", expec_value) 
            #print("expected value = ", reversed_evolved_probe)
            #print("expectation value: ", reversed_evolved_probe)
            print_loc(global_print_loc)
            return expec_value
 
        elif ham_dim > ham_minus_dim:
            log_print([" Dimensions don't match; IQLE not applicable"], log_file=log_file, log_identifier=log_identifier)
            return 0.5
            """
            dim = true_dim
            smaller_dim = sim_dim
            probe = random_probe(dim)
            qt_probe = qt.Qobj(trim_vector(probe, final_num_qubits=smaller_dim))
            #print("qt probe: ", qt_probe)
            to_keep = range(smaller_dim)
            evolved_probe = evolved_state(ham=ham_true, t=1, state=probe)
            evolved_qt_obj = qt.Qobj(evolved_probe)
            evolved_qt_obj.dims = [[2]*dim, [1]*dim]
            evolved_partial_trace = evolved_qt_obj.ptrace(to_keep)[:]
            sim_unitary = h.exp_ham(ham_sim, t, plus_or_minus=1)
            sim_unitary_dagger = sim_unitary.conj().T # U_adjoint
            Rho_U = np.dot(evolved_partial_trace, sim_unitary)
            U_adjoint_Rho_U = qt.Qobj(np.dot(sim_unitary_dagger, Rho_U))
            U_adjoint_Rho_U.dims = [[2]*smaller_dim, [2]*smaller_dim]
            #print("U_rho_U = ", U_adjoint_Rho_U)
            expected_value = qt.expect(U_adjoint_Rho_U, qt_probe)
            #print("expected value = ", expected_value)
            #print("expectation value: ", expected_value)
            return expected_value
            """
        else: 
            log_print(["giving expectation value = 0.5 because simulated system is bigger than true system."], log_file=log_file, log_identifier=log_identifier)
            return 0.5
    else: 
        log_print(["Implement trotterization in IQLE evolve function (Evo.py)"], log_file=log_file, log_identifier=log_identifier)
 
 
