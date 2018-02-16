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
 
 
 
def get_pr0_array_qle(t_list, modelparams, oplist, probe, use_exp_custom=True, enable_sparse=True, ham_list=None):
    
    print_loc(global_print_loc)
    num_particles = len(modelparams)

    #if num_particles==1: print("True probe \n", probe)
    #else: print("Simulated probe \n", probe)
    num_times = len(t_list)
    output = np.empty([num_particles, num_times])
    for evoId in range(num_particles): ## todo not sure about length/arrays here
#        ham = np.tensordot(modelparams[evoId], oplist, axes=1)
        for tId in range(len(t_list)):
            ham = np.tensordot(modelparams[evoId], oplist, axes=1)
            t = t_list[tId]
            print_loc(global_print_loc)
            
            output[evoId][tId] = expectation_value(ham=ham, t=t, state=probe, use_exp_custom=use_exp_custom, enable_sparse=enable_sparse)
            if output[evoId][tId] < 0:
                print("[QLE] Negative probability : \t \t probability = ", output[evoId][tId])
            elif output[evoId][tId] > 1.000000000000001: ## todo some times getting p=1.0 show up
                print("[QLE] Probability > 1: \t \t probability = ", output[evoId][tId]) 
            #print("(i,j) = (", evoId, tId,") \t val: ", output[evoId][tId])
    print_loc(global_print_loc) 
    return output
 

def get_pr0_array_iqle(t_list, modelparams, oplist, ham_minus, probe, use_exp_custom=True, enable_sparse=True, trotterize=True, ham_list = None):
    print_loc(global_print_loc)
    num_particles = len(modelparams)
    num_times = len(t_list)
     
 
    output = np.empty([num_particles, num_times])
 
    if print_pr0: print("output has shape ", output.shape)
 
    for evoId in range( output.shape[0]): ## todo not sure about length/arrays here
        ham = np.tensordot(modelparams[evoId], oplist, axes=1)
        for tId in range(len(t_list)):
            t = t_list[tId]
             
            output[evoId][tId] = iqle_evolve(ham = ham, ham_minus = ham_minus, t=t, probe=probe, use_exp_custom=use_exp_custom, enable_sparse=enable_sparse)
            print_loc(global_print_loc)
            if output[evoId][tId] < 0:
                print("negative probability : \t \t probability = ", output[evoId][tId])
            elif output[evoId][tId] > 1.000000000000001:
                print("[IQLE] Probability > 1: \t \t probability = ", output[evoId][tId]) 
            #print("(i,j) = (", evoId, tId,") \t val: ", output[evoId][tId])
    print_loc(global_print_loc)
    return output

     
     
     
## Partial trace functionality
 
def expectation_value(ham, t, state=None, choose_random_probe=False, use_exp_custom=True, enable_sparse=True):
# todo: list of probes, maybe 5 is enough? test with different values
    print_loc(global_print_loc)
    print_loc(global_print_loc)
    if choose_random_probe is True: 
        num_qubits = int(np.log2(np.shape(ham)[0]))
        state = random_probe(num_qubits)
    elif random_probe is False and state is None: 
        print ("expectation value function: you need to either pass a state or set choose_random_probe=True")
    #print("\n")
    print_loc(global_print_loc)
    #import hamiltonian_exponentiation as h
    #u_psi_exp_custom = h.unitary_evolve(ham, t, state, enable_sparse_functionality=enable_sparse)
    #u_psi_linalg = evolved_state(ham, t, state, use_exp_custom=False)
    
    #diff = u_psi_exp_custom - u_psi_linalg
    
    #print("diff in expec val from linalg to exp custom = ", diff, "\t abs=", np.absolute(diff))
    
    
    if use_exp_custom and ham_exp_installed:
      import hamiltonian_exponentiation as h    
      u_psi = h.unitary_evolve(ham, t, state, enable_sparse_functionality=enable_sparse)
    else:
      u_psi = evolved_state(ham, t, state, use_exp_custom=False)
    print_loc(global_print_loc)
    
    probe_bra = state.conj().T
    
    
    psi_u_psi = np.dot(probe_bra, u_psi)
    
    print_loc(global_print_loc) 
    expec_value = np.abs(psi_u_psi)**2 ## TODO MAKE 100% sure about this!!

    print_loc(global_print_loc)
#    del u_psi
    print_loc(global_print_loc)
    
    if np.abs(expec_value) > 1.0000001:
        print("expectation value function has value ", np.abs(psi_u_psi**2))
        print("t=", t, "\nham = \n ", ham, "\n probe : \n", state, "\n probe normalisation : ", np.linalg.norm(u_psi))
    print_loc(global_print_loc)
    print_expec_value_intermediate = False
    if print_expec_value_intermediate:
      print("Bra : \n", probe_bra)
      print("u_psi\n", u_psi)
      
      print("psi_psi:\n", np.dot(probe_bra, state))
      print("t=", t)
      print("Ham : \n", ham)
      print("probe : \n", state)
      print("u_psi: \n", u_psi)
      print("psi_u_psi: \n", psi_u_psi)
      
      print("Expectation value : ", expec_value)
    return expec_value
 
def evolved_state(ham, t, state, use_exp_custom=True, enable_sparse=True):
    #import hamiltonian_exponentiation as h
    from scipy import linalg
    print_loc(global_print_loc)
  
    #print("Enable sparse : ", enable_sparse)    
    if use_exp_custom and ham_exp_installed:
      unitary = h.exp_ham(ham, t, enable_sparse_functionality=enable_sparse)
    else:
      print("Note: using linalg for exponentiating Hamiltonian.")
      unitary = linalg.expm(-1j*ham*t)
    print_loc(global_print_loc)
    ev_state = np.dot(unitary, state)
    del unitary # to save space
    print_loc(global_print_loc)
    return ev_state


 
def random_probe(num_qubits):
    dim = 2**num_qubits
    real = np.random.rand(1,dim)
    imaginary = np.random.rand(1,dim)
    complex_vectors = np.empty([1, dim])
    complex_vectors = real +1.j*imaginary
    norm_factor = np.linalg.norm(complex_vectors)
    probe = complex_vectors/norm_factor
    return probe[0][:]
 
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
 
def iqle_evolve(ham, ham_minus, t, probe, trotterize=True, use_exp_custom=True, enable_sparse=True ):
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
            print(" Dimensions don't match; IQLE not applicable")
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
            print("giving expectation value = 0.5 because simulated system is bigger than true system.")
            return 0.5
    else: 
        print("Implement trotterization in IQLE evolve function (Evo.py)")
 
 
