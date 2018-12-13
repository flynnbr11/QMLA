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
Useful functions
""" 
def full_model_string(operations):
    """
    operations must be a dict with elements:
    - 'dim' : number of qubits
    - 'terms' : list of lists of tuple of the form, 
        e.g. [ (1, 'x'), (2, 'y')]
        i.e. tuples (qubit_id, pauli_operator)
        Each nested list gives a term, which are all added together for the full model
    Reconstructs unique model name for that Hamiltonian. 
    """
    
    
    # Note TODO: this doesn't give an error when tuples are 
    # given which aren't used. it should
    from DataBase import alph
    terms = operations['terms']
    num_qubits = operations['dim']
    num_terms = len(terms)
    all_terms = []
    for i  in range(len(terms)):
        single_term = terms[i]
        single_term_dict = dict(single_term)

        model_name = ''

        t_str=''
        for i in range(1, 1+num_qubits):
            if i in single_term_dict:
                pauli = single_term_dict[i]
            else:
                pauli ='i'
            t_str+='T'
            if i==num_qubits:
                t_str = ''
            model_name += str(pauli + t_str)

        all_terms.append(model_name)

    all_terms = list(set(list(all_terms)))
    p_str = 'P'
    for i in range(num_qubits-1):
        p_str+='P'

    full_model = p_str.join(all_terms)    
    # full_model = DataBase.alph(full_model)
    full_model = alph(full_model)
    return full_model

def operations_dict_from_name(mod_name):
    constituents = DataBase.get_constituent_names_from_name(mod_name)
    num_qubits = DataBase.get_num_qubits(mod_name)
    initial_t_str = ''
    all_terms = []
    for j in range(num_qubits-1):
        initial_t_str += 'T' 

    for i in range(len(constituents)):
        t_str = initial_t_str
        single_term = constituents[i]
        all_tuples_this_term = []
        n_minus_1_qubit_operators = single_term
        for k in range(num_qubits):
            if len(t_str) > 0:
                split_by_nth_qubit = n_minus_1_qubit_operators.split(t_str)
                this_tuple = (num_qubits-k, split_by_nth_qubit[1])
                n_minus_1_qubit_operators = split_by_nth_qubit[0]
                t_str = t_str[:-1]
            else:
                this_tuple = (num_qubits-k, n_minus_1_qubit_operators)

            all_tuples_this_term.append(this_tuple)

        all_tuples_this_term = sorted(all_tuples_this_term)
        all_terms.append(all_tuples_this_term)

    operations = {
        'dim' : num_qubits, 
        'terms' : all_terms
    }
    
    return operations


def make_term_transverse(term, transverse_axis):
    dimension = DataBase.get_num_qubits(term)
        
    transverse_terms = []
    op_dict = operations_dict_from_name(term)
    
    for i in range(1, 1+dimension):
        transverse_term = [(i, transverse_axis)]
        op_dict['terms'].append(transverse_term)
    term_with_transverse_components = full_model_string(op_dict)
    return term_with_transverse_components



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
        latex_term+='S_{'+r_terms+'}'
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
    num_qubits=2, 
    return_branch_dict='latex_terms', 
    **kwargs
):
    from UserFunctions import max_spawn_depth_info
    num_qubits = max_spawn_depth_info['non_interacting_ising'] + 1
    num_qubits=1 # deliberately cause crash to inspect error. 
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



############# Model to Branch convention functions ##########


def branch_is_num_params(latex_mapping_file, **kwargs):
    print(
        "branch_is_num_params", 
        latex_mapping_file
    )
    with open(latex_mapping_file) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 

    latex_name_map = {}
    for c in content:
        this_tuple = eval(c)
        model_string = this_tuple[0]
        latex_name = this_tuple[1]
        latex_name_map[model_string] = latex_name    # this mapping assigns models to branches with the number of parameters they have
    
    model_names = list(set(list(latex_name_map.keys())))
    num_params_by_mod = {}
    model_branches = {}

    for mod in model_names:
        num_params = len(DataBase.get_constituent_names_from_name(mod))
        num_params_by_mod[mod] = num_params
        latex_name = latex_name_map[mod]
        model_branches[latex_name] = num_params

    return model_branches


def branch_is_num_dims(latex_mapping_file, **kwargs):
    with open(latex_mapping_file) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 

    latex_name_map = {}
    for c in content:
        this_tuple = eval(c)
        model_string = this_tuple[0]
        latex_name = this_tuple[1]
        latex_name_map[model_string] = latex_name    # this mapping assigns models to branches with the number of parameters they have
    
    model_names = list(set(list(latex_name_map.keys())))
    model_branches = {}

    for mod in model_names:
        # num_params = len(DataBase.get_constituent_names_from_name(mod))
        num_qubits = DataBase.get_num_qubits(mod)
        latex_name = latex_name_map[mod]
        model_branches[latex_name] = num_qubits

    return model_branches

def interacting_ising_nearest_neighbours_all_names(
    # growth_generator,
    **kwargs
):
    import ModelGeneration
    all_models = []

    initial_models = ['xTx', 'yTy', 'zTz']
    num_qubits = 5

    models_on_branches = {
        2 : initial_models
    }

    model_list = initial_models
    for i in range(2, num_qubits+1):
        new_models = ModelGeneration.interacting_nearest_neighbour_ising(model_list)
        all_models.extend(new_models)
        model_list = new_models
        models_on_branches[i] = new_models
    
    
    
    return all_models

def nearest_neighbour_ising_latex_name(
    name,
    **kwargs
):
    num_qubits = DataBase.get_num_qubits(name)

    paulis = ['x', 'y', 'z']

    for p in paulis:
        if p in name:
            core_pauli = p

    latex_rep = str(
        '$' + 
         core_pauli +   
        '^{'+
        '\\otimes' +
        str(num_qubits) +
        '}$'
    )
    return latex_rep
    
def hubbard_latex(name):
    individual_terms = DataBase.get_constituent_names_from_name(name)
    latex_term = ''
    for term in individual_terms:
        if term[0] == 'h':
            split_term = term.split('_')
            sites = []
            for st in split_term:
                if st[0] not in ['h', 'd']:
                    sites.append(st)
            latex_this_term = str(
                'H_{' + 
                sites[0] + 
                ',' + sites[1] + 
                '}'
            )
            latex_term += latex_this_term
        else:
            latex_this_term = interaction_latex_name(term)
            list_letters = list(latex_this_term)
            while '$' in list_letters:
                list_letters.remove('$')
            new_latex_term = ''
            for l in list_letters:
                new_latex_term += l
            latex_term += new_latex_term
        
    latex_term = str('$' + latex_term + '$')
    return latex_term

def interaction_latex_name(
    name, 
    interacting_term = '\sigma', 
    **kwargs
):
    name = DataBase.alph(name)
    op_dict = operations_dict_from_name(name)
    terms = op_dict['terms']
    num_terms = len(terms)
    full_model_string = ''
    all_term_strings= []

    for term in terms:
        this_term_string = '' 
        paulis = []
        qubits = []

        for j in range(len(term)):
            action = term[j]
            if action[1] != 'i':
                qubits.append(action[0])
                paulis.append(action[1])

        if len(paulis) != len(qubits):
            print("unmatching number of qubits to pauli operators in naming function.")

        else:
            this_term_string += interacting_term
            this_term_string += '^{'

            for p in paulis:
                this_term_string += str(p)
            this_term_string += '}'

            this_term_string += '_{'
            for q in qubits:
                this_term_string += str(q)
            this_term_string += '}'

            all_term_strings.append(this_term_string)

    all_term_strings = sorted(all_term_strings)
    final_string = ''.join(all_term_strings)
    final_string = str('$' + final_string + '$')
    return final_string
