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


#######################################


"""
Assign each generation rule a latex name mapping
"""
latex_naming_functions = {
	None : default_latex_wrapping,
	'two_qubit_ising_rotation_hyperfine' : latex_name_ising, 
	'two_qubit_ising_rotation' : latex_name_ising, 
}



def get_latex_name(
    name, 
    growth_generator=None,
    **kwargs
):
	try:
		# if mapping doesn't work, default to just wrap in $__$. 
		latex_mapping = latex_naming_functions[growth_generator]
		latex_representation = latex_mapping(name, **kwargs)
	except:
		latex_mapping = latex_naming_functions[None]
		latex_representation = latex_mapping(name, **kwargs)
	# print("Latex Mapping used", latex_mapping)

	return latex_representation
