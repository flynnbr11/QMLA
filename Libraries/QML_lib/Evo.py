from __future__ import print_function # so print doesn't show brackets
import qinfer as qi
import numpy as np
import scipy as sp
import inspect
import time
import sys as sys
import os as os
from MemoryTest import print_loc
sys.path.append((os.path.join("..")))

global_print_loc = False 
use_linalg = False
use_sparse = False 
 
try: 
    import hamiltonian_exponentiation as h
    # TODO set to true after testing
    ham_exp_installed = True
     
except:
    ham_exp_installed = False
 
if (use_linalg): 
    # override and use linalg.expm even if hamiltonian_exponentiation is installed
    ham_exp_installed = False
     
# Generic states and Pauli matrices  

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
     
     
# Functions for evolution 
def getH(_pars, _ops):
    return (np.tensordot(_pars, _ops, axes=1))[0]
 

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
 
 
def get_pr0_array_qle(t_list, modelparams, oplist, probe,
        use_experimental_data=False, use_exp_custom=True,
        exp_comparison_tol=None, enable_sparse=True, 
        ham_list=None, log_file='QMDLog.log', log_identifier=None
):
    from rq import timeouts
    print_loc(global_print_loc)
    num_particles = len(modelparams)
    num_times = len(t_list)
    output = np.empty([num_particles, num_times])
    for evoId in range(num_particles): ## todo not sure about length/arrays here
        for tId in range(len(t_list)):
            ham = np.tensordot(modelparams[evoId], oplist, axes=1)

            """
            # Log print to prove True Hamiltonian is time dependent.
            if num_particles == 1: 
                log_print(
                    [
                    "Time dependent, true Hamiltonian:\n", ham
                    ], 
                    log_file, log_identifier
                )
            """
                
            t = t_list[tId]
            print_loc(global_print_loc)
            try:
                
                if use_experimental_data == True:
                    # TODO replace call to exp_val with call to Hahn exp val
                    output[evoId][tId] = hahn_evolution(
                        ham = ham,
                        t = t,
                        state = probe,
                        log_file = log_file, 
                        log_identifier = log_identifier
                    )
                    
                else: 
                    output[evoId][tId] = expectation_value(ham=ham, t=t, state=probe,
                        use_exp_custom=use_exp_custom, 
                        compare_exp_fncs_tol=exp_comparison_tol,
                        enable_sparse=enable_sparse, log_file=log_file,
                        log_identifier=log_identifier
                    )
            except NameError:
                log_print(["Error raised; unphysical expecation value.",
                    "\nHam:\n", ham,
                    "\nt=", t,
                    "\nState=", probe
                    ], 
                    log_file, log_identifier
                )
                sys.exit()
            except timeouts.JobTimeoutException:
                log_print(["RQ Time exception. \nprobe=", probe,
                    "\nt=", t,"\nHam=", ham], log_file, log_identifier
                )
                sys.exit()
#                raise
                
            if output[evoId][tId] < 0:
                log_print(["[QLE] Negative probability : \t \t probability = ", 
                    output[evoId][tId]], log_file, log_identifier
                )
            elif output[evoId][tId] > 1.001: 
                log_print(["[QLE] Probability > 1: \t \t probability = ", 
                    output[evoId][tId]], log_file, log_identifier
                )
    return output
 

def get_pr0_array_iqle(t_list, modelparams, oplist, ham_minus, 
        probe, use_exp_custom=True, enable_sparse=True, 
        exp_comparison_tol=None, trotterize=True, ham_list = None,
        log_file='QMDLog.log',log_identifier=None
):
    print_loc(global_print_loc)
    num_particles = len(modelparams)
    num_times = len(t_list)
    output = np.empty([num_particles, num_times])
 
    for evoId in range( output.shape[0]): ## todo not sure about length/arrays here
        ham = np.tensordot(modelparams[evoId], oplist, axes=1)
        for tId in range(len(t_list)):
            t = t_list[tId]
             
            output[evoId][tId] = iqle_evolve(ham = ham, 
                ham_minus = ham_minus, t=t, probe=probe,
                use_exp_custom=use_exp_custom,
                compare_exp_fncs_tol=exp_comparison_tol,
                enable_sparse=enable_sparse, log_file=log_file,
                log_identifier=log_identifier
            )
            if output[evoId][tId] < 0:
                log_print(["negative probability : \t \t probability = ",
                    output[evoId][tId]], log_file, log_identifier
                )
            elif output[evoId][tId] > 1.000000000000001:
                log_print(["[IQLE] Probability > 1: \t \t probability = ",
                    output[evoId][tId]], log_file, log_identifier
                )
    return output

     
     
     
## Partial trace functionality
 
def expectation_value(ham, t, state=None, choose_random_probe=False,
    use_exp_custom=True, enable_sparse=True, print_exp_details=False,
    exp_fnc_cutoff=20, compare_exp_fncs_tol=None, log_file='QMDLog.log',
    log_identifier=None, debug_plot_print=False
):

    if choose_random_probe is True: 
        num_qubits = int(np.log2(np.shape(ham)[0]))
        state = random_probe(num_qubits)
    elif random_probe is False and state is None: 
        log_print(["expectation value function: you need to \
            either pass a state or set choose_random_probe=True"],
            log_file=log_file, log_identifier=log_identifier
        )
    
    if compare_exp_fncs_tol is not None: # For testing custom ham-exp function
        u_psi_linalg = evolved_state(ham, t, state, 
            use_exp_custom=False, print_exp_details=print_exp_details,
            exp_fnc_cutoff=exp_fnc_cutoff
        )
        u_psi_exp_custom = evolved_state(ham, t, state,
            use_exp_custom=True, print_exp_details=print_exp_details,
            exp_fnc_cutoff=exp_fnc_cutoff
        )
        
        diff = np.max(np.abs(u_psi_linalg-u_psi_exp_custom))
        if (np.allclose(u_psi_linalg, u_psi_exp_custom, 
            atol=compare_exp_fncs_tol) == False
        ):
            log_print(["Linalg/ExpHam give different evolved state by", diff],
                log_file=log_file, log_identifier=log_identifier
            )
            u_psi = u_psi_linalg
        else:
            u_psi = u_psi_exp_custom
            
    else: # compute straight away; don't compare exponentiations
        if use_exp_custom and ham_exp_installed: 
            try:
              u_psi = evolved_state(ham, t, state, use_exp_custom=True,
                  print_exp_details=print_exp_details, 
                  exp_fnc_cutoff=exp_fnc_cutoff
              )
            except ValueError:
                log_print(["Value error when exponentiating Hamiltonian. Ham:\n",
                    ham, "\nProbe: ", state], log_file=log_file,
                    log_identifier=log_identifier
                )
        else:
            u_psi = evolved_state(ham, t, state, use_exp_custom=False,
                print_exp_details=print_exp_details, 
                exp_fnc_cutoff=exp_fnc_cutoff
            )
    
    probe_bra = state.conj().T
    try:
        psi_u_psi = np.dot(probe_bra, u_psi)
    except UnboundLocalError: 
        log_print(
            [
            "UnboundLocalError when exponentiating Hamiltonian. t=", t, 
            "\nHam:\n", ham,
            "\nProbe: ", state
            ], log_file=log_file,
            log_identifier=log_identifier
        )
        raise
    
    expec_value = np.abs(psi_u_psi)**2 ## TODO MAKE 100% sure about this!!
    
    expec_value_limit=1.10000000001 # maximum permitted expectation value
    
    if expec_value > expec_value_limit:
        log_print(["expectation value function has value ", 
            np.abs(psi_u_psi**2)], log_file=log_file, 
            log_identifier=log_identifier
        )
        log_print(["t=", t, "\nham = \n ", ham, "\nprobe : \n", state, 
            "\nprobe normalisation:", np.linalg.norm(state), "\nU|p>:", 
            u_psi, "\nnormalisation of U|p>:", np.linalg.norm(u_psi), 
            "\n<p|U|p>:", psi_u_psi, "\nExpec val:", expec_value],
            log_file=log_file, log_identifier=log_identifier
        )
        log_print(["Recalculating expectation value using linalg."],
            log_file=log_file, log_identifier=log_identifier
        )
        u_psi = evolved_state(ham, t, state, 
            use_exp_custom=False, log_file=log_file, 
            log_identifier=log_identifier
        )
        psi_u_psi = np.dot(probe_bra, u_psi)
        expec_value = np.abs(psi_u_psi)**2 ## TODO MAKE 100% sure about this!!
        raise NameError('UnphysicalExpectationValue') 
    
    return expec_value


 
def evolved_state(ham, t, state, use_exp_custom=True, 
    enable_sparse=True, print_exp_details=False, exp_fnc_cutoff=10, log_file=None,
    log_identifier=None
):
    #import hamiltonian_exponentiation as h
    from scipy import linalg
    print_loc(global_print_loc)
  
    if t>1e6: ## Try limiting times to use to 1 million
        import random
        t = random.randint(1e6, 3e6) # random large number but still computable without error
    
#        t=1e6 # TODO PUT BACK IN. testing high t to find bug. 

    if use_exp_custom and ham_exp_installed:
        if log_file is not None:
            log_print(
                ["Using custom expm. Exponentiating\nt=",t, "\nHam=\n", ham],
                log_file, log_identifier
            )
        unitary = h.exp_ham(ham, t, enable_sparse_functionality=enable_sparse,
            print_method=print_exp_details, scalar_cutoff=t+1
        )
    else:
        if log_file is not None:
            iht = (-1j*ham*t)
            log_print(["Using linalg.expm. Exponentiating\nt=",t, "\nHam=\n",
                ham, "\n-iHt=\n", iht, "\nMtx elements type:", 
                type(iht[0][0]), "\nMtx type:", type(iht)], 
                log_file, log_identifier
            )
        unitary = linalg.expm(-1j*ham*t)
        
        if log_file is not None:
            log_print(["linalg.expm gives \nU=\n",unitary], log_file, log_identifier)
    
    ev_state = np.dot(unitary, state)

    if log_file is not None:
        log_print(["evolved state fnc. Method details printed in worker log. \nt=",
            t, "\nHam=\n", ham, "\nprobe=", state, "\nU=\n", unitary, "\nev_state=",
            ev_state], log_file, log_identifier
        )
    del unitary # to save space
    return ev_state


def traced_expectation_value_project_one_qubit_plus(
    ham,
    t, 
    state, 
):
    """ 
    Expectation value tracing out all but 
    first qubit to project onto plus state
    """
    import qutip 
    import numpy as np
    
    one_over_sqrt_two = 1/np.sqrt(2) + 0j 
    plus = np.array([one_over_sqrt_two, one_over_sqrt_two])
    one_qubit_plus = qutip.Qobj(plus)
    
    ev_state = evolved_state(ham, t, state)
    qstate = qutip.Qobj(ev_state)
    qstate.dims= [[2,2], [1,1]]
    traced_state = qstate.ptrace(0) # TODO: to generalise, make this exclude everything apart from 0th dimension ?
    expect_value = np.abs(qutip.expect(traced_state, one_qubit_plus))**2
    
    return expect_value
    

# Expecactation value function using Hahn inversion gate:

def hahn_evolution(ham, t, state, log_file=None, log_identifier=None):
    
    #hahn_angle = np.pi/2
    #hahn = np.kron(hahn_angle*sigmaz(), np.eye(2))
    #inversion_gate = linalg.expm(-1j*hahn)
    # inversion gate generated as above, done once and hardocded since this doesn't change.
    inversion_gate = np. array([
        [0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j]]
    )

    unitary_time_evolution = h.exp_ham(ham, t)

    total_evolution = np.dot(
        unitary_time_evolution,
        np.dot(inversion_gate,
              unitary_time_evolution)
    )

    evolved_state = np.dot(total_evolution, state)
    state_bra = state.conj().T

    psi_u_psi = np.dot(state_bra, evolved_state)
    expec_value = np.abs(psi_u_psi)**2
    
    if expec_value > 1.0000000001:
        if log_file is not None: 
            log_print([
                "(Hahn) Expec value:2, expec_value",
                "\nHam:\n", ham,
                "\nState=\n", state,
                "\nt=", t,
                ],
                log_file = log_file,
                log_identifier = log_identifier
            )
                
    
    return expec_value


 
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
 
def iqle_evolve(ham, ham_minus, t, probe, 
    trotterize=True, use_exp_custom=True, enable_sparse=True, 
    log_file='QMDLog.log', log_identifier=None
):
    print_loc(global_print_loc)
    ham_dim = int(np.log2(np.shape(ham)[0])) 
    ham_minus_dim = int(np.log2(np.shape(ham_minus)[0]))
 
 
    if trotterize == True: 
        if ham_dim == ham_minus_dim: 
            H = ham_minus - ham ##reversed because exp_ham function calculated e^{-iHt}
            print_loc(global_print_loc)
            expec_value = expectation_value(H, t, state=probe,
                use_exp_custom=use_exp_custom, enable_sparse=enable_sparse
            )
            print_loc(global_print_loc)
            return expec_value
 
        elif ham_dim > ham_minus_dim:
            log_print([" Dimensions don't match; IQLE not applicable"],
                log_file=log_file, log_identifier=log_identifier
            )
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
            log_print(["giving expectation value = 0.5 because simulated \
                system is bigger than true system."], log_file=log_file,
                log_identifier=log_identifier
            )
            return 0.5
    else: 
        log_print(["Implement trotterization in IQLE evolve function (Evo.py)"],
            log_file=log_file, log_identifier=log_identifier
        )
 
 
