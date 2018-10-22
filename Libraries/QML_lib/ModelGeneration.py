"""
Functions for generation of new models.
"""

from __future__ import print_function # so print doesn't show brackets

import numpy as np
import itertools as itr

import os as os
import sys as sys 
import pandas as pd
import warnings
import time as time
import Evo as evo
import DataBase 
import warnings

global paulis_list
# paulis_list = {'i' : np.eye(2), 'x' : evo.sigmax(), 'y' : evo.sigmay(), 'z' : evo.sigmaz()}

# Dict of max spawn depths, corresponding to different generation functions. 

##################################################################################
##################################################################################
"""
Essential functions. Functions below are specific, for generating terms according to given rules. new_model_list is the wrapper function which calls the selected generator rule.
"""
##################################################################################
##################################################################################

def time_seconds():
    import datetime
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    return time

def log_print(to_print_list, log_file):
    identifier = str(str(time_seconds()) +" [MOD_GEN]")
    if type(to_print_list)!=list:
        to_print_list = list(to_print_list)

    print_strings = [str(s) for s in to_print_list]
    to_print = " ".join(print_strings)
    with open(log_file, 'a') as write_log_file:
        print(identifier, str(to_print), file=write_log_file, flush=True)

"""

max_spawn_depth_info = {
    'qhl_TEST' : 2, 
    'simple_ising' : 1,
    'hyperfine' : 3,
    'ising_non_transverse' : 5,
    'ising_transverse' : 11,
    'hyperfine_like' : 8,
    'two_qubit_ising_rotation' : 2,
    'two_qubit_ising_rotation_hyperfine' : 5,
    'two_qubit_ising_rotation_hyperfine_transverse' : 8,
    'test_multidimensional' : 10
}



def max_spawn_depth(generator, log_file):
    if generator not in max_spawn_depth_info:
        log_print(["Generator not recognised; does not have maximum spawn depth or generation function"], log_file)
    else:
        return max_spawn_depth_info[generator]

def new_model_list(model_list, spawn_depth, model_dict, log_file, 
    options=['x', 'y', 'z'], generator='simple_ising'
):
    log_print(["Generating new models according to best of last round: ",
        model_list, "; and options:", options], log_file
    )
    if generator == 'simple_ising':
        return simple_ising(generator_list=model_list, options=options)
    elif generator=='ising_non_transverse':
        return ising_non_transverse(model_list, spawn_step=spawn_depth)
    elif generator == 'ising_transverse':
        return ising_transverse(model_list, spawn_step=spawn_depth)
    elif generator == 'two_qubit_ising_rotation_hyperfine':
        return hyperfine_like(model_list, spawn_step=spawn_depth,
        model_dict=model_dict, log_file=log_file
    )
    elif generator == 'two_qubit_ising_rotation_hyperfine_transverse':
        return hyperfine_like(model_list, spawn_step=spawn_depth,
        model_dict=model_dict, log_file=log_file
    )
    elif generator == 'hyperfine_like':
        return hyperfine_like(model_list, spawn_step=spawn_depth,
        model_dict=model_dict, log_file=log_file
    )
    elif generator == 'test_multidimensional':
        return test_multidimensional(
            model_list=model_list, 
            spawn_step=spawn_depth, 
            log_file=log_file
        )
    else:
        log_print(["Generator", generator, "not recognised"], log_file)        

"""



##################################################################################
##################### Model Generation Functions ############################################
##################################################################################



"""
Functions for generation of random model names for testing/debugging.
"""
        
def random_model_name(num_dimensions=1, num_terms=1):
    """
    Return a valid (simple) model name of given number of dimensions and terms. 
    """
    import random
    paulis = ['x', 'y', 'z', 'i']
    p_str = ''
    t_str = ''
    
    for i in range(num_dimensions):
        p_str += 'P'
    for j in range(num_dimensions -1):
        t_str += 'T'
    
    
    summed_term = ''
    for j in range(1, num_terms+1):
        this_term = generate_term(num_dimensions)
        summed_term += this_term
        if(j!=num_terms):
            summed_term += p_str
    
    
    # Don't allow returning just identity in any dimension #TODO?
    while summed_term == ('i' or 'iTi' or 'iTiTTi' or 'iTiTTiTTTi' or
        'iTiTTiTTTiTTTTi' or 'iTiTTiTTTiTTTTiTTTTTi' or 
        'iTiTTiTTTiTTTTiTTTTTiTTTTTTi' or 
        'iTiTTiTTTiTTTTiTTTTTiTTTTTTiTTTTTTTi'
    ):
        summed_term = random_model_name(num_dimensions, num_terms) 
    
    return summed_term


def random_ising_chain(num_dimensions=1, include_identity = True, num_directions = 1):
    """
    Return a valid (simple) model name of given number of dimensions and terms. 
    """
    import random
    paulis_full = ['x', 'y', 'z']
    if include_identity: paulis_full.append('i')

    if num_directions > len(paulis_full):
        paulis = paulis_full
        warnings.warn("Number of dimensions exceeded the Pauli ops space")
    else:      
        paulis = random.sample(paulis_full, num_directions)    

    if num_dimensions == 1:
        return random.choice(paulis)

    num_terms = num_dimensions-1    
    
    p_str = ''
    t_str = ''
    
    for i in range(num_dimensions):
        p_str += 'P'
    for j in range(num_dimensions -1):
        t_str += 'T'
    
    
    summed_term = ''
    for j in range(1, num_terms+1):
        this_term = generate_term(num_dimensions, paulis)
        summed_term += this_term
        if(j!=num_terms):
            summed_term += p_str
            
    return summed_term



def generate_term(num_dimensions, paulis=['x', 'y', 'z', 'i']):
    """
    For use only in random_model_name() function. 
    """
    import random
    t_str = ''
    running_str =''
    
    if num_dimensions == 1:
        return random.choice(paulis)
    else:
        for j in range(num_dimensions):
            t_str += 'T'
            running_str += random.choice(paulis)
            
            if j != num_dimensions -1:
                running_str += t_str

        return running_str
            
def single_pauli_multiple_dim(num_dimensions, paulis=['x', 'y', 'z', 'i'], pauli=None):
    """
    For use only in random_model_name() function. 
    """
    import random
    t_str = ''
    running_str =''
    
    if pauli is None: pauli=random.choice(paulis)

    if num_dimensions == 1:
        return random.choice(paulis)
    else:
        for j in range(num_dimensions):
            t_str += 'T'
            running_str += pauli
            
            if j != num_dimensions -1:
                running_str += t_str

        return running_str



"""
Specific Hamiltonian generators. 
Return strings corresponding to model names following naming convention.
Use these strings with DataBase.operator class.
"""

def interaction_ham(qubit_list, operator_list, num_qubits):
    
    t_str = ''
    running_str = ''
    for i in range(num_qubits):
        t_str += 'T'
        
        if i in qubit_list:
            running_str += operator_list[qubit_list.index(i)]
            
        else: 
            running_str += 'i'
        if len(t_str)!=num_qubits:
            running_str += t_str
    return running_str

def ising_fully_interacting(num_qubits):
    # fully interacting Ising Hamiltonian name
    this_idx = 0
    max_idx = 0
    for i in range(num_qubits):
        max_idx += i ## Triangle number of 
    p_str = ''
    for a in range(num_qubits):
        p_str += 'P'
    
    running_str=''
    for i in range(num_qubits):
        for j in range(i, num_qubits):
            if i!=j:
                this_idx+=1
                op_list = ['z', 'z']
                qub_list = [i,j]
                new_term = interaction_ham(qubit_list=qub_list,
                    operator_list=op_list, num_qubits=num_qubits
                )
                running_str += new_term
                if(this_idx < max_idx):
                    running_str += p_str
    return running_str


## Manipulate input model names 

def identity_interact(subsystem, num_qubits, return_operator=False):
    new_string = ''
    sub_dim = DataBase.get_num_qubits(subsystem)
    if sub_dim >= num_qubits:
        if return_operator: 
            return op
        else: 
            return op.name
    
    for i in range(sub_dim, num_qubits):
        t_str = 'T'
        for j in range(i-1):
            t_str += 'T'
        new_string += t_str+'i'
    output_string = subsystem + new_string
    if return_operator: 
        return DataBase.operator(output_string)
    else:
        return output_string
        

def dimensionalise_name_by_name(name, true_dim, return_operator=False):
    op = DataBase.operator(name)
    constituents = op.constituents_names
    new_list=[]

    for term in constituents: 
        new_list.append(identity_interact(subsystem=term, num_qubits=true_dim))

    p_str = ''
    for i in range(true_dim):
        p_str+='P'    

    new_name = p_str.join(new_list)
    
    if return_operator: 
        return DataBase.operator(new_name)
    else: 
        return new_name

        
        
def dimensionalise_name_by_operator(op, true_dim, return_operator=False):
    constituents = op.constituents_names
    new_list=[]

    for term in constituents: 
        new_list.append(identity_interact(subsystem=term, num_qubits=true_dim))

    p_str = ''
    for i in range(true_dim):
        p_str+='P'    

    new_name = p_str.join(new_list)
    
    if return_operator: 
        return DataBase.operator(new_name)
    else: 
        return new_name


def dimensionalise_name_by_name_list(constituents, true_dim, return_operator=False):
    new_list=[]

    for term in constituents: 
        new_list.append(identity_interact(subsystem=term, num_qubits=true_dim))

    p_str = ''
    for i in range(true_dim):
        p_str+='P'    

    new_name = p_str.join(new_list)
    if return_operator: 
        return DataBase.operator(new_name)
    else: 
        return new_name
            
            
def simple_ising(
    generator_list, 
    options=['x', 'y', 'z'],
    **kwargs
):
    new_options = []

    for gen in generator_list: 
        num_qubits = DataBase.get_num_qubits(gen)
        t_str = ''
        for a in range(num_qubits):
            t_str+='T'
        for opt in options:
            new_options.append(gen+t_str+opt)
    
    return new_options


### spawn function to match process followed during experimental QMD ###

def single_pauli_multiple_dim(
    num_qubits, 
    log_file, 
    paulis=['x', 'y', 'z', 'i'], 
    pauli=None
):
    import random
    t_str = ''
    running_str =''
    
    if pauli is None: pauli=random.choice(paulis)

    if num_qubits == 1:
        return random.choice(paulis)
    else:
        for j in range(num_qubits):
            t_str += 'T'
            running_str += pauli
            
            if j != num_qubits -1:
                running_str += t_str

        return running_str
        
def ising_non_transverse(
    model_list, 
    spawn_step, 
    log_file,
    **kwargs
):
    single_qubit_terms = ['xTi', 'yTi', 'zTi']
    nontransverse_terms = ['xTx', 'yTy', 'zTz']

    if len(model_list) > 1:
        log_print(["Only one model required for non-transverse Ising growth."],
            log_file
        )
        return False
    else:
        model = model_list[0]
    
    present_terms = model.split('PP')

    new_models = []
    if spawn_step in [1,2]:
        for term in single_qubit_terms:
            if term not in present_terms:
                new_model = model+'PP'+term
                new_models.append(new_model)

    elif spawn_step in [3,4,5]:
        for term in nontransverse_terms:
            if term not in present_terms:
                new_model = model+'PP'+term
                new_models.append(new_model)
                
    return new_models    


def ising_transverse(
    model_list, 
    spawn_step, 
    log_file,
    **kwargs
):
# TODO before using this function, need to add a max_spawn_depth to the dict above for ising_transverse. How many spawns can it support?
    single_qubit_terms = ['xTi', 'yTi', 'zTi']
    nontransverse_terms = ['xTx', 'yTy', 'zTz']
    all_transverse_terms = ['xTy', 'xTz', 'yTx', 'yTz', 'zTy', 'zTx']
    transverse_terms = ['xTy', 'xTz','yTz']
    all_two_qubit_terms = ( single_qubit_terms
        + nontransverse_terms  + transverse_terms
    )
     
    if len(model_list) > 1:
        log_print(["Only one model required for transverse Ising growth."],
            log_file
        )
        return False
    else:
        model = model_list[0]

    present_terms = model.split('PP')

    new_models = []
    if spawn_step in [1,2]:
        for term in single_qubit_terms:
            if term not in present_terms:
                new_model = model+'PP'+term
                new_models.append(new_model)

    elif spawn_step in [3,4,5]:
        for term in nontransverse_terms:
            if term not in present_terms:
                new_model = model+'PP'+term
                new_models.append(new_model)

    else:
        for term in transverse_terms:
            if term not in present_terms:
                new_model = model+'PP'+term
                new_models.append(new_model)

    return new_models    

def hyperfine_like(
    model_list, 
    spawn_step, 
    model_dict, 
    log_file,
    **kwargs
):
# TODO before using this function, need to add a max_spawn_depth to the dict above for ising_transverse. How many spawns can it support?
    import random
    single_qubit_terms = ['xTi', 'yTi', 'zTi']
    nontransverse_terms = ['xTx', 'yTy', 'zTz']
    transverse_terms = ['xTy', 'xTz', 'yTz']
    all_two_qubit_terms = ( single_qubit_terms + nontransverse_terms
        + transverse_terms
    )
    if len(model_list) > 1:
        log_print(["Only one model required for transverse Ising growth."],
            log_file
        )
        return False
    else:
        model = model_list[0]

    present_terms = model.split('PP')

    new_models = []
    if spawn_step in [1,2]:
        for term in single_qubit_terms:
            if term not in present_terms:
                new_model = model+'PP'+term
                new_models.append(new_model)
    elif spawn_step in [3,4,5]:
        for term in nontransverse_terms:
            if term not in present_terms:
                new_model = model+'PP'+term
                new_models.append(new_model)

    elif spawn_step == 6: 
        i=0
        while i < 3:
            term = random.choice(transverse_terms)
            
            if term not in present_terms:
                new_model = model+'PP'+term
                if ( 
                    DataBase.check_model_in_dict(new_model, model_dict) == False
                    and new_model not in new_models
                ):
                    
                    new_models.append(new_model)
                    i+=1
    elif spawn_step == 7: 
        i=0
        while i < 2:
            term = random.choice(transverse_terms)
            
            if term not in present_terms:
                new_model = model+'PP'+term
                if (
                    DataBase.check_model_in_dict(new_model, model_dict) == False
                    and new_model not in new_models
                ):
                    
                    new_models.append(new_model)
                    i+=1

    elif spawn_step == 8: 
        i=0
        while i < 1:
            term = random.choice(transverse_terms)
            
            if term not in present_terms:
                new_model = model+'PP'+term
                if (
                    DataBase.check_model_in_dict(new_model, model_dict) == False
                    and new_model not in new_models
                ):
                    
                    new_models.append(new_model)
                    i+=1
    return new_models    
        

def test_multidimensional(
    model_list, 
    spawn_step, 
    log_file,
    **kwargs
):
    
    # a test generation rule to check if QMD still works across dimensions. 

    new_models = []
    
    for m in model_list:
        dim = DataBase.get_num_qubits(m)
        num_terms = len(DataBase.get_constituent_names_from_name(m))
        for i in range(2):
            new_mod = random_model_name(dim, num_terms+1)
            new_models.append(new_mod)
    
    log_print(
        [
        'Input models:', 
        model_list,
        'New models:', 
        new_models
        ],
        log_file = log_file
    )

    if spawn_step==max_spawn_depth_info['test_multidimensional']:
        log_print(
            [
            'Multidensional test at spawn step=', 
            spawn_step
            ],
            log_file
        )

    return new_models


def p_t_actions(name):
    num_qubits = DataBase.get_num_qubits(name)

    p_str = 'P'
    t_str = ''

    for i in range(num_qubits-1):
        p_str+='P'
        t_str+='T'

    actions = {
        'p' : p_str,
        't' : t_str
    }

    return actions    



def existing_branch_champs_test(    
    model_list, 
    spawn_step, 
    model_dict, 
    log_file,
    current_champs
):
    new_models = []
    
    for name in model_list:
        actions = p_t_actions(model_list)
        p_str = actions['p']
        one_qubit_terms = ['x', 'y', 'z']
        individual_terms = name.split(p_str)
        
        remaining_terms = list( set(one_qubit_terms) - set(individual_terms) )
        if spawn_step < 3 : 
            for term in remaining_terms:
                new_mod = str(name)
                new_mod += str(p_str + term)
                new_models.append(new_mod)
        else: 
            # current_champs.append('xPyPzPi')
            return current_champs
            
    return new_models


def non_interacting_ising(model_list, **kwargs):
    new_models = []
    paulis = ['x', 'y', 'z']
    for mod in model_list:
        t_str = p_t_actions(mod)['t']
        new_t_str = str( t_str + 'T')
        for p in paulis:
            new_model = str(mod + new_t_str + p)        
            new_models.append(new_model)
    return new_models



# ##### Wrapper function and dict

# model_generation_functions = {
#     'simple_ising' : simple_ising,
#     'ising_non_transverse' : ising_non_transverse,
#     'ising_transverse' : ising_transverse,
#     'two_qubit_ising_rotation_hyperfine' : hyperfine_like,
#     'two_qubit_ising_rotation_hyperfine_transverse' : hyperfine_like,
#     'hyperfine_like' : hyperfine_like,
#     'test_multidimensional' : test_multidimensional,
# }

# def new_model_generator(generator, **kwargs):
#     model_func = model_generation_functions[generator]
#     print("Using model generation function:", model_func)
#     return model_func(**kwargs)


##################################################################################
##################### Tree Finished Functions ############################################
##################################################################################


def spawn_depth_check(
    generator,
    max_spawn_depth_info,
    spawn_step, 
):
    if spawn_step == max_spawn_depth_info[generator]:
        return True 
    else:
        return False



