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
paulis_list = {'i' : np.eye(2), 'x' : evo.sigmax(), 'y' : evo.sigmay(), 'z' : evo.sigmaz()}


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
    while summed_term == ('i' or 'iTi' or 'iTiTTi' or 'iTiTTiTTTi' or 'iTiTTiTTTiTTTTi' or 'iTiTTiTTTiTTTTiTTTTTi' or 'iTiTTiTTTiTTTTiTTTTTiTTTTTTi' or 'iTiTTiTTTiTTTTiTTTTTiTTTTTTiTTTTTTTi'):
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
                new_term = interaction_ham(qubit_list=qub_list, operator_list=op_list, num_qubits=num_qubits)
                running_str += new_term
                if(this_idx < max_idx):
                    running_str += p_str
    return running_str


## Manipulate input model names 

def identity_interact(subsystem, num_qubits, return_operator=False):
    new_string = ''
#    op = DataBase.operator(subsystem)
#    sub_dim=op.num_qubits
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
            
            
def new_model_list_grow_one_qubit(generator_list, options=['x', 'y', 'z']):
    new_options = []

    for gen in generator_list: 
        num_qubits = DataBase.get_num_qubits(gen)
        t_str = ''
        for a in range(num_qubits):
            t_str+='T'
        for opt in options:
            new_options.append(gen+t_str+opt)
    
    return new_options
    
    
def new_model_list(model_list, options=['x', 'y', 'z'], generator='simple_ising'):
    if generator == 'simple_ising':
        return new_model_list_grow_one_qubit(generator_list=model_list, options=options)
        #todo integrate Andreas' simple Ising growth
