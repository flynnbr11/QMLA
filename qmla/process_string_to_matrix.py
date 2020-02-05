import numpy as np
import itertools as itr
import warnings
import os as os
import sys as sys
import pandas as pd
import warnings
import copy
import time as time

import fermilib
from fermilib.ops import FermionOperator
from fermilib.transforms import get_sparse_operator

from qmla import database_framework
import qmla.model_naming as model_naming
import qmla.logging

__all__ = [
    'process_basic_operator'
]

def log_print(
    to_print_list, 
    log_file
):
    qmla.logging.print_to_log(
        to_print_list = to_print_list, 
        log_file = log_file,
        log_identifier = 'Model Generation'
    )


def process_basic_operator(basic_operator):
    # # from qmla import model_generation
    # from qmla import process_string_to_matrix   

    if basic_operator[0:1] == 'h_':
        # import model_generation
        mtx = process_hubbard_operator(
            basic_operator
        )
        # hopping_matrix(basic_operator)
    elif '1Dising' in basic_operator:
        # import model_generation
        mtx = process_1d_ising(
            basic_operator
        )
    elif 'Heis' in basic_operator:
        # import model_generation
        mtx = process_heisenberg_xyz(
            basic_operator
        )
    elif 'nv' in basic_operator:
        # import model_generation
        mtx = process_n_qubit_NV_centre_spin(
            basic_operator
        )
    elif 'pauliSet' in basic_operator:
        mtx = process_multipauli_term(
            term=basic_operator
        )
    elif 'transverse' in basic_operator:
        mtx = process_transverse_term(
            term=basic_operator
        )
    elif 'FH' in basic_operator:
        # print("[DB] Processing hopping operator:", term)
        mtx = process_fermi_hubbard_term(
            term=basic_operator
        )
    else:
        mtx = database_framework.core_operator_dict[basic_operator]

    return mtx



def process_transverse_term(term):
    # terms of form transverse_x_d3
    # transverse matrix is a single matrix of form,.e.g
    # XII + IXI + IIX
    # where num qubits=3, transverse axis=X

    components = term.split('_')
    components.remove('transverse')
    core_operators = list(sorted(database_framework.core_operator_dict.keys()))

    for l in components:
        if l[0] == 'd':
            dim = int(l.replace('d', ''))
        elif l in core_operators:
            transverse_axis = l
    mtx = transverse_axis_matrix(
        num_qubits=dim,
        transverse_axis=transverse_axis
    )
    return mtx


def process_multipauli_term(term):
    # term of form pauliSet_aJb_iJk_dN
    # where a is operator on site i
    # b is operator on site k
    # N is total number of sites
    # e.g. pauliSet_xJy_1J3_d4

    components = term.split('_')
    components.remove('pauliSet')
    core_operators = list(sorted(database_framework.core_operator_dict.keys()))
    for l in components:
        if l[0] == 'd':
            dim = int(l.replace('d', ''))
        elif l[0] in core_operators:
            operators = l.split('J')
        else:
            sites = l.split('J')
    # get strings when splitting the list elements
    sites = [int(s) for s in sites]
    # want tuples of (site, operator) for dict logic
    all_terms = list(zip(sites, operators))

    term_dict = {
        'dim': dim,
        'terms': [all_terms]
    }

    full_mod_str = model_naming.full_model_string(term_dict)
    return database_framework.compute(full_mod_str)


def process_n_qubit_NV_centre_spin(term):
    components = term.split('_')
    for l in components:
        if l[0] == 'd':
            dim = int(l.replace('d', ''))
        elif l == 'spin':
            term_type = 'spin'
        elif l == 'interaction':
            term_type = 'interaction'
        elif l in ['x', 'y', 'z']:
            pauli = l

    if term_type == 'spin':
        t_str = 'T'
        op_name = str(pauli)

        for d in range(dim - 1):
            op_name += str(t_str + 'i')
            t_str += 'T'
    elif term_type == 'interaction':
        p_str = 'P' * dim
        op_name = ''
        for d in range(dim - 1):
            t_str = 'T'
            single_term_name = str(pauli)
            for j in range(dim - 1):
                single_term_name += str(t_str)
                if d == j:
                    single_term_name += pauli
                else:
                    single_term_name += 'i'
                t_str += 'T'
            op_name += single_term_name
            if d < (dim - 2):
                op_name += p_str

    # print("Type {} ; name {}".format(term_type, op_name))
    return database_framework.compute(op_name)

def process_1d_ising(term):
    components = term.split('_')
    components.remove('1Dising')
    include_transverse_component = include_chain_component = False

    for l in components:
        if l[0] == 'd':
            dim = int(l.replace('d', ''))
        elif l[0] == 'i':
            chain_axis = str(l.replace('i', ''))
            include_chain_component = True
        elif l[0] == 't':
            include_transverse_component = True
            transverse_axis = str(l.replace('t', ''))

    if include_chain_component == True:
        return ising_interaction_component(
            num_qubits=dim,
            interaction_axis=chain_axis
        )

    elif include_transverse_component == True:
        return ising_transverse_component(
            num_qubits=dim,
            transverse_axis=transverse_axis
        )


def transverse_axis_matrix(
    num_qubits,
    transverse_axis
):
    individual_transverse_terms = []
    for i in range(1, 1 + num_qubits):
        single_term = ''
        t_str = 'T'
        for q in range(1, 1 + num_qubits):
            if i == q:
                single_term += transverse_axis
            else:
                single_term += 'i'

            if q != num_qubits:
                single_term += t_str
                t_str += 'T'

        individual_transverse_terms.append(single_term)
    running_mtx = database_framework.compute(individual_transverse_terms[0])
    for term in individual_transverse_terms[1:]:
        running_mtx += database_framework.compute(term)
    return running_mtx


def ising_transverse_component(
    num_qubits,
    transverse_axis
):
    return transverse_axis_matrix(num_qubits, transverse_axis)

def ising_interaction_component(num_qubits, interaction_axis):

    individual_interaction_terms = []

    for i in range(1, num_qubits):
        single_term = ''
        t_str = 'T'
        for q in range(1, num_qubits + 1):
            if i == q or i + 1 == q:

                single_term += interaction_axis
            else:
                single_term += 'i'

            if q != num_qubits:
                single_term += t_str
                t_str += 'T'

        individual_interaction_terms.append(single_term)

    running_mtx = database_framework.compute(individual_interaction_terms[0])

    for term in individual_interaction_terms[1:]:
        running_mtx += database_framework.compute(term)

    return running_mtx

def process_heisenberg_xyz(term):
    components = term.split('_')
    components.remove('Heis')
    include_transverse_component = include_chain_component = False

    for l in components:
        if l[0] == 'd':
            dim = int(l.replace('d', ''))
        elif l[0] == 'i':
            chain_axis = str(l.replace('i', ''))
            include_chain_component = True
        elif l[0] == 't':
            include_transverse_component = True
            transverse_axis = str(l.replace('t', ''))

    if include_chain_component == True:
        return single_axis_nearest_neighbour_interaction_chain(
            num_qubits=dim,
            interaction_axis=chain_axis
        )

    elif include_transverse_component == True:
        return single_axis_transverse_component(
            num_qubits=dim,
            transverse_axis=transverse_axis
        )
def single_axis_nearest_neighbour_interaction_chain(
    num_qubits,
    interaction_axis='x'
):

    individual_interaction_terms = []

    for i in range(1, num_qubits):
        single_term = ''
        t_str = 'T'
        for q in range(1, num_qubits + 1):
            if i == q or i + 1 == q:

                single_term += interaction_axis
            else:
                single_term += 'i'

            if q != num_qubits:
                single_term += t_str
                t_str += 'T'

        individual_interaction_terms.append(single_term)

    running_mtx = database_framework.compute(individual_interaction_terms[0])

    for term in individual_interaction_terms[1:]:
        running_mtx += database_framework.compute(term)

    return running_mtx

def single_axis_transverse_component(
    num_qubits,
    transverse_axis='z'
):

    individual_transverse_terms = []

    for i in range(1, 1 + num_qubits):
        single_term = ''
        t_str = 'T'
        for q in range(1, 1 + num_qubits):
            if i == q:
                single_term += transverse_axis
            else:
                single_term += 'i'

            if q != num_qubits:
                single_term += t_str
                t_str += 'T'

        individual_transverse_terms.append(single_term)

    running_mtx = database_framework.compute(individual_transverse_terms[0])

    for term in individual_transverse_terms[1:]:
        running_mtx += database_framework.compute(term)

    return running_mtx


def process_fermi_hubbard_chemical(constituents):
    # constituents ~ ['dN', 'i'], N = num sites, i = site index for onsite
    # term
    for c in constituents:
        if c[0] == 'd':
            num_sites = int(c[1:])
        else:
            site_number = int(c)

    dimensional_description = "{}".format(2 * num_sites - 1)
    dimensional_fermion_op = FermionOperator(dimensional_description)

    # index with respect to basis (site_number, spin_type)
    i = 2 * site_number - 2
    down_term = FermionOperator(((i, 0), ))
    up_term = FermionOperator(((i, 1), ))
    down_term += dimensional_fermion_op
    up_term += dimensional_fermion_op

    mtx = jordan_wigner_mtx(up_term) + jordan_wigner_mtx(down_term) - \
        2 * jordan_wigner_mtx(dimensional_fermion_op)
    return np.array(mtx)


def process_fermi_hubbard_onsite(constituents):
    # constituents ~ ['dN', 'i'], N = num sites, i = site index for onsite
    # term
    for c in constituents:
        if c[0] == 'd':
            num_sites = int(c[1:])
        else:
            site_number = int(c)
    dimensional_description = "{}".format(2 * num_sites - 1)
    dimensional_fermion_op = FermionOperator(dimensional_description)

    # index with respect to basis (site_number, spin_type)
    i = 2 * site_number - 2
    num_term = FermionOperator(((i, 1), (i, 0), (i + 1, 1), (i + 1, 0)))
    # operator of form {c^{\dagger}_i c_i c^{\dagger}_{i+1} c_{i+1}}
    num_term += dimensional_fermion_op

    mtx = jordan_wigner_mtx(num_term) - \
        jordan_wigner_mtx(dimensional_fermion_op)
    return np.array(mtx)


def process_fermi_hubbard_hopping(constituents):
    for c in constituents:
        if c in ['down', 'up']:
            spin_type = c
        elif c[0] == 'd':
            num_sites = int(c[1:])
        else:
            sites = [int(s) for s in c.split('h')]

    i_idx = 2 * (sites[0] - 1) - 2  # 2i -2
    j_idx = 2 * (sites[1] - 1) - 2  # 2j -2
    if spin_type == 'down':
        i_idx = 2 * (sites[0] - 1)
        j_idx = 2 * (sites[1] - 1)
    elif spin_type == 'up':
        i_idx = 2 * (sites[0]) - 1
        j_idx = 2 * (sites[1]) - 1

    dimensional_description = "{}".format(2 * num_sites - 1)
    dimensional_fermion_op = FermionOperator(dimensional_description)

    hopping_term = FermionOperator(((i_idx, 1), (j_idx, 0)))
    hopping_term += FermionOperator(((j_idx, 1), (i_idx, 0)))
    hopping_term += dimensional_fermion_op

    mtx = jordan_wigner_mtx(hopping_term) - \
        jordan_wigner_mtx(dimensional_fermion_op)
    return np.array(mtx)
    
def jordan_wigner_mtx(fermion_operator):
    """
    Calls fermilib functinoality to compute complete matrix given a
    FermionOperator class fermion.
    """
    return fermilib.transforms.get_sparse_operator(fermion_operator).todense()


def process_fermi_hubbard_term(term):
    #     term ~ FHhop_ihj_s_dN, FHonsite_i_dN, FHchemical_i_dN:
    #     (i,j) sites to hop between; s spin type to hop ('up' or 'down')
    #     i site to count; N number sites total

    constituents = term.split('_')
    for c in constituents:
        if c == 'FHhop':
            constituents.remove(c)
            mtx = process_fermi_hubbard_hopping(constituents)
        elif c == 'FHonsite':
            constituents.remove(c)
            mtx = process_fermi_hubbard_onsite(constituents)
        elif c == 'FHchemical':
            constituents.remove(c)
            mtx = process_fermi_hubbard_chemical(constituents)
    return mtx

def process_hubbard_operator(
    term
):
    # TODO deprecated?
    # for use in computing base level terms in a model, used in database_framework.
    return base_hubbard_grouped_term(term)
