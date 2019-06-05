from __future__ import print_function # so print doesn't show brackets
import qinfer as qi
import numpy as np
import scipy as sp
import inspect
import time
import sys as sys
import os as os
from MemoryTest import print_loc
import ExpectationValues
# from UserFunctions import expectation_value_wrapper
import UserFunctions
sys.path.append((os.path.join("..")))

global_print_loc = False 
use_linalg = False
use_sparse = False 

global test_growth_class_implementation
test_growth_class_implementation = True
 
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
 
 
def get_pr0_array_qle(
        t_list, modelparams, 
        oplist, probe,
        measurement_type='full_access',
        growth_class=None, 
        use_experimental_data=False, use_exp_custom=True,
        exp_comparison_tol=None, enable_sparse=True, 
        ham_list=None, 
        log_file='QMDLog.log', log_identifier=None
):
    from rq import timeouts
    print_loc(global_print_loc)
    num_particles = len(modelparams)
    num_times = len(t_list)
    output = np.empty([num_particles, num_times])
    # print(
    #     "[Evo] pr0 array",
    #     "\n\t modelparams:", modelparams, 
    #     "\n\t oplist:", oplist, 
    # )

    for evoId in range(num_particles): ## todo not sure about length/arrays here

        try:
            ham = np.tensordot(
                modelparams[evoId], oplist, axes=1
            )
        except:
            log_print(
                [
                "Failed to build Hamiltonian.",
                "\nmodelparams:", modelparams[evoId], 
                "\noplist:", oplist
                ], 
                log_file, log_identifier
            )
            raise
        outputs_this_ham = {}
        unique_times_considered_this_ham = []

        for tId in range(len(t_list)):
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
            # print("[EVO]t=", t)
            # if t in unique_times_considered_this_ham:
            #     output[evoId][tId] = outputs_this_ham[t]
            #     print("saved calc")
            # else:
            try:
                # log_print(
                #     [
                #     "[getpr0] EvoID=", evoId,
                #     "\n\tState=", probe
                #     ], 
                #     log_file, log_identifier
                # )
                try:
                    likel = growth_class.expectation_value(
                        ham = ham,
                        t = t,
                        state = probe,
                        log_file = log_file, 
                        log_identifier = log_identifier
                    )
                except:         
                    if test_growth_class_implementation == True: raise
                    likel = UserFunctions.expectation_value_wrapper(
                    # output[evoId][tId] = ExpectationValues.expectation_value_wrapper(
                        method=measurement_type,
                        ham = ham,
                        t = t,
                        state = probe,
                        log_file = log_file, 
                        log_identifier = log_identifier
                    )
                output[evoId][tId] = likel
                # unique_times_considered_this_ham.append(t)
                # outputs_this_ham[t] = likel

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
                log_print(
                    [
                        "[QLE] Negative probability : \
                        \t \t probability = ", 
                        output[evoId][tId]
                    ], 
                    log_file, log_identifier

                )
            elif output[evoId][tId] > 1.001: 
                log_print(
                    [
                        "[QLE] Probability > 1: \
                        \t \t probability = ", 
                        output[evoId][tId]
                    ], 
                    log_file, 
                    log_identifier
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
     