"""
Functions for naming, and retrieving latex representation
of models according to QMD naming convention.
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



"""
First define all the functions which are callable 
by the wrappers for converting name strings to 
latex representations
"""
def default_latex_wrapping(name):
    return str('$'+str(name) + '$')


def latex_name_ising(name):
    # TODO generalise this 
    if name=='x' or name=='y' or name=='z':
        return '$'+name+'$'

    num_qubits = DataBase.get_num_qubits(name)
    terms=name.split('PP')
    rotations = ['xTi', 'yTi', 'zTi']
    hartree_fock = ['xTx', 'yTy', 'zTz']
    transverse = ['xTy', 'xTz', 'yTz']
    
    
    present_r = []
    present_hf = []
    present_t = []
    
    for t in terms:
        if t in rotations:
            present_r.append(t[0])
        elif t in hartree_fock:
            present_hf.append(t[0])
        elif t in transverse:
            string = t[0]+t[-1]
            present_t.append(string)
        # else:
        #     print("Term",t,"doesn't belong to rotations, Hartree-Fock or transverse.")
        #     print("Given name:", name)
    present_r.sort()
    present_hf.sort()
    present_t.sort()

    r_terms = ','.join(present_r)
    hf_terms = ','.join(present_hf)
    t_terms = ','.join(present_t)
    
    
    latex_term = ''
    if len(present_r) > 0:
        latex_term+='R_{'+r_terms+'}'
    if len(present_hf) > 0:
        latex_term+='HF_{'+hf_terms+'}'
    if len(present_t) > 0:
        latex_term+='T_{'+t_terms+'}'
    


    final_term = '$'+latex_term+'$'
    if final_term != '$$':
        return final_term

    else:
        plus_string = ''
        for i in range(num_qubits):
            plus_string+='P'
        individual_terms = name.split(plus_string)
        individual_terms = sorted(individual_terms)

        latex_term = '+'.join(individual_terms)
        final_term = '$'+latex_term+'$'
        return final_term




##### All possible model list functions ######
def ising_terms_rotation_hyperfine(return_branch_dict=None):
    pauli_terms = ['x','y','z']

    branches= {}
    branch_by_term_dict = {}
    for i in range(9):
        branches[i] = []
    
    rotation_terms = []
    hf_terms = []
    transverse_terms = []

    for t in pauli_terms:
        rotation_terms.append(t+'Ti')
        hf_terms.append(t+'T'+t)
        for k in pauli_terms:
            if k>t:
                transverse_terms.append(t+'T'+k)

    ising_terms = []            
    add = 'PP'

    for r in rotation_terms:
        ising_terms.append(r)
        branches[0].append(r)
        branch_by_term_dict[r] = 0
        
    for r in rotation_terms:
        new_terms=[]
        for i in rotation_terms:
            if r<i:
                branches[1].append(r+add+i)
                branch_by_term_dict[r+add+i] = 1
                new_terms.append(r+add+i)
        ising_terms.extend(new_terms)

    full_rotation = add.join(rotation_terms)
    ising_terms.append(full_rotation)
    branches[2].append(full_rotation)
    branch_by_term_dict[full_rotation] = 2
    

    for t in hf_terms:
        new_term = full_rotation+add+t
        branches[3].append(new_term)
        branch_by_term_dict[new_term] = 3
        ising_terms.append(new_term)

    for t in hf_terms:
        for k in hf_terms:
            if t<k:
                dual_hf_term= full_rotation+add+t+add+k
                branches[4].append(dual_hf_term)
                branch_by_term_dict[dual_hf_term] = 4
                ising_terms.append(dual_hf_term)

    for t in hf_terms:
        for l in hf_terms:
            for k in hf_terms:
                if t<k<l:
                    triple_hf = full_rotation + add + t + add + k + add + l
                    branches[5].append(triple_hf)
                    ising_terms.append(triple_hf)
                    branch_by_term_dict[triple_hf] = 5

    
    # latex_terms = [DataBase.latex_name_ising(i) for i in ising_terms]
    latex_terms = [
        # get_latex_name(
        #     #  in this case this list is only generating ising type so hard code growth generator
        #     name=i, growth_generator='two_qubit_ising_rotation_hyperfine_transverse'
        # ) for i in ising_terms
        latex_name_ising(name=i) for i in ising_terms
    ]
    
    if return_branch_dict=='branches':
        return branches
    elif return_branch_dict=='terms':
        return branch_by_term_dict
    elif return_branch_dict=='latex_terms':
        return latex_terms
    elif return_branch_dict=='term_branch_dict':
        for k in list(branch_by_term_dict.keys()):
            # branch_by_term_dict[DataBase.latex_name_ising(k)]=(
            branch_by_term_dict[
                # get_latex_name(
                #     #  in this case this list is only generating ising type so hard code growth generator
                #     name = k, growth_generator='two_qubit_ising_rotation_hyperfine_transverse')
                # ] = (branch_by_term_dict.pop(k)
                latex_name_ising(name = k)] = (branch_by_term_dict.pop(k)
            )
        return branch_by_term_dict
    else:
        return latex_terms




def ising_terms_full_list(return_branch_dict=None):
    pauli_terms = ['x','y','z']

    branches= {}
    branch_by_term_dict = {}
    for i in range(9):
        branches[i] = []
    
    rotation_terms = []
    hf_terms = []
    transverse_terms = []

    for t in pauli_terms:
        rotation_terms.append(t+'Ti')
        hf_terms.append(t+'T'+t)
        for k in pauli_terms:
            if k>t:
                transverse_terms.append(t+'T'+k)

    ising_terms = []            
    add = 'PP'

    for r in rotation_terms:
        ising_terms.append(r)
        branches[0].append(r)
        branch_by_term_dict[r] = 0
        
    for r in rotation_terms:
        new_terms=[]
        for i in rotation_terms:
            if r<i:
                branches[1].append(r+add+i)
                branch_by_term_dict[r+add+i] = 1
                new_terms.append(r+add+i)
        ising_terms.extend(new_terms)

    full_rotation = add.join(rotation_terms)
    ising_terms.append(full_rotation)
    branches[2].append(full_rotation)
    branch_by_term_dict[full_rotation] = 2
    

    for t in hf_terms:
        new_term = full_rotation+add+t
        branches[3].append(new_term)
        branch_by_term_dict[new_term] = 3
        ising_terms.append(new_term)

    for t in hf_terms:
        for k in hf_terms:
            if t<k:
                dual_hf_term= full_rotation+add+t+add+k
                branches[4].append(dual_hf_term)
                branch_by_term_dict[dual_hf_term] = 4
                ising_terms.append(dual_hf_term)

    for t in hf_terms:
        for l in hf_terms:
            for k in hf_terms:
                if t<k<l:
                    triple_hf = full_rotation + add + t + add + k + add + l
                    branches[5].append(triple_hf)
                    ising_terms.append(triple_hf)
                    branch_by_term_dict[triple_hf] = 5



    for t in transverse_terms:
        transverse_term= triple_hf+add+t
        branches[6].append(transverse_term)
        branch_by_term_dict[transverse_term] = 6
        ising_terms.append(transverse_term)


    for t in transverse_terms:
        for k in transverse_terms:
            if t<k:
                dual_transverse_term= triple_hf+add+t+add+k
                branches[7].append(dual_transverse_term)
                branch_by_term_dict[dual_transverse_term] = 7
                ising_terms.append(dual_transverse_term)

    for t in transverse_terms:
        for l in transverse_terms:
            for k in transverse_terms:
                if t<k<l:
                    triple_hf_term= triple_hf+add+t+add+k+add+l
                    branch_by_term_dict[triple_hf_term] = 8
                    branches[8].append(triple_hf_term)
                    ising_terms.append(triple_hf_term)

    
    latex_terms = [
        # DataBase.latex_name_ising(i) for i in ising_terms
        # get_latex_name(
        #     name=i, 
        #     #  in this case this list is only generating ising type so hard code growth generator
        #     growth_generator='two_qubit_ising_rotation_hyperfine_transverse' 
        # ) for i in ising_terms
        latex_name_ising(name=i) for i in ising_terms
    ]
    
    if return_branch_dict=='branches':
        return branches
    elif return_branch_dict=='terms':
        return branch_by_term_dict
    elif return_branch_dict == 'latex_terms':
        return latex_terms
    elif return_branch_dict == 'term_branch_dict':
        for k in list(branch_by_term_dict.keys()):
            # branch_by_term_dict[DataBase.latex_name_ising(k)]=(
            branch_by_term_dict[
            	# get_latex_name(
	            #     name=k, 
	            #     growth_generator='two_qubit_ising_rotation_hyperfine_transverse'
            	# )
            	latex_name_ising(name=k)
            	] = (
            		branch_by_term_dict.pop(k)
            )
        return branch_by_term_dict
        
    else:
        return latex_terms




#######################################


# """
# Assign each generation rule a latex name mapping
# """
# latex_naming_functions = {
# 	None : default_latex_wrapping,
# 	'two_qubit_ising_rotation_hyperfine' : 
# 		latex_name_ising, 
# 	'two_qubit_ising_rotation_hyperfine_transverse' : 
# 		latex_name_ising, 
# }

# all_models_functions = {
# 	None : ising_terms_rotation_hyperfine,
# 	'two_qubit_ising_rotation_hyperfine' : 
# 		ising_terms_rotation_hyperfine, 
# 	'two_qubit_ising_rotation_hyperfine_transverse' : 
# 		ising_terms_full_list, 
# }


# ######
# ### ----- Wrapper functions. ----- ###
# ######

# def get_latex_name(
#     name, 
#     growth_generator=None,
#     **kwargs
# ):
# 	try:
# 		# if mapping doesn't work, default to just wrap in $__$. 
# 		latex_mapping = latex_naming_functions[growth_generator]
# 		latex_representation = latex_mapping(name, **kwargs)
# 	except:
# 		latex_mapping = latex_naming_functions[None]
# 		latex_representation = latex_mapping(name, **kwargs)
# 	# print("Latex Mapping used", latex_mapping)

# 	return latex_representation


# def get_all_model_names(
# 	growth_generator = None,
# 	**kwargs
# ):
# 	all_models = all_models_functions[growth_generator]
# 	# print("all model function:", all_models)
# 	return all_models(**kwargs)


def test_return_champs_ALL_MODELS(**kwargs):
	models = ['x', 'y', 'z', 'xPy', 'xPz', 'yPz', 'xPyPz']

	return models
	

def non_interacting_model(core_pauli, num_qubits):
    
    t_str = ''
    model = core_pauli
    for i in range(num_qubits):
        t_str+='T'
        model += str(t_str +core_pauli)
        
    return model


def non_interacting_ising_all_names(
    num_qubits=5, 
    return_branch_dict='latex_terms', 
    **kwargs
):
    paulis = ['x', 'y', 'z']
    all_models=[]
    models_on_branches = {0: ['']}
    t_str = ''
    for i in range(num_qubits):
        models_on_branches[i+1] = []
        for m in models_on_branches[i]:
            for p in paulis:
                new_mod = str(m + t_str + p)
                models_on_branches[i+1].append(new_mod)
                all_models.append(new_mod)
        t_str += 'T'

    models_on_branches.pop(0)
    terms_by_branch = {}
    for b in models_on_branches:
        for v in list(models_on_branches[b]):
            latex_term = default_latex_wrapping(v)
            terms_by_branch[latex_term] = b

    latex_terms = [
        default_latex_wrapping(i)
        for i in all_models
    ]

    try:
        if return_branch_dict=='branches':
            return models_on_branches
        elif return_branch_dict=='terms':
            return all_models_latex
        elif return_branch_dict=='latex_terms':
            return latex_terms
        elif return_branch_dict == 'term_branch_dict':
            return terms_by_branch
    except:
        return latex_terms
