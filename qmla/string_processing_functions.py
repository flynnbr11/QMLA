
import numpy as np

from fermilib.ops import FermionOperator
import fermilib.transforms

from qmla import model_building_utilities

import qmla.logging

__all__ = [
    "process_n_qubit_NV_centre_spin", 
    "process_multipauli_term",
    "process_likewise_pauli_sum", 
    "process_fermi_hubbard_term", 
]

def log_print(
    to_print_list,
    log_file
):
    r"""Wrapper for :func:`~qmla.print_to_log`"""
    qmla.logging.print_to_log(
        to_print_list=to_print_list,
        log_file=log_file,
        log_identifier='Model Generation'
    )

######################
# Section: construction facilities
######################


def full_model_string(operations):
    """
    operations must be a dict with elements:
    - 'dim' : number of qubits
    - 'terms' : list of lists of tuple of the form,
        e.g. [[ (1, 'x'), (2, 'y')]]
        i.e. tuples (qubit_id, pauli_operator)
        Each nested list gives a term, which are all added together for the full model
    Reconstructs unique model name for that Hamiltonian.
    """

    # Note TODO: this doesn't give an error when tuples are
    # given which aren't used. it should
    from qmla.model_building_utilities import alph
    terms = operations['terms']
    num_qubits = operations['dim']
    num_terms = len(terms)
    all_terms = []
    for i in range(len(terms)):
        single_term = terms[i]
        single_term_dict = dict(single_term)

        model_name = ''

        t_str = ''
        for i in range(1, 1 + num_qubits):
            if i in single_term_dict:
                pauli = single_term_dict[i]
            else:
                pauli = 'i'
            t_str += 'T'
            if i == num_qubits:
                t_str = ''
            model_name += str(pauli + t_str)

        all_terms.append(model_name)

    all_terms = list(set(list(all_terms)))
    p_str = 'P'
    for i in range(num_qubits - 1):
        p_str += 'P'

    full_model = '+'.join(all_terms)
    full_model = alph(full_model)
    return full_model


def operations_dict_from_name(mod_name):
    constituents = model_building_utilities.get_constituent_names_from_name(mod_name)
    num_qubits = model_building_utilities.get_num_qubits(mod_name)
    initial_t_str = ''
    all_terms = []
    for j in range(num_qubits - 1):
        initial_t_str += 'T'

    for i in range(len(constituents)):
        t_str = initial_t_str
        single_term = constituents[i]
        all_tuples_this_term = []
        n_minus_1_qubit_operators = single_term
        for k in range(num_qubits):
            if len(t_str) > 0:
                split_by_nth_qubit = n_minus_1_qubit_operators.split(t_str)
                this_tuple = (num_qubits - k, split_by_nth_qubit[1])
                n_minus_1_qubit_operators = split_by_nth_qubit[0]
                t_str = t_str[:-1]
            else:
                this_tuple = (num_qubits - k, n_minus_1_qubit_operators)

            all_tuples_this_term.append(this_tuple)

        all_tuples_this_term = sorted(all_tuples_this_term)
        all_terms.append(all_tuples_this_term)

    operations = {
        'dim': num_qubits,
        'terms': all_terms
    }

    return operations


######################
# Section: process individual terms
######################

def process_transverse_term(term):
    # terms of form transverse_x_d3
    # transverse matrix is a single matrix of form,.e.g
    # XII + IXI + IIX
    # where num qubits=3, transverse axis=X

    components = term.split('_')
    components.remove('transverse')
    core_operators = list(sorted(model_building_utilities.core_operator_dict.keys()))

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
    print("PROCESSING MULTIPAULI")

    components = term.split('_')
    components.remove('pauliSet')
    core_operators = list(sorted(model_building_utilities.core_operator_dict.keys()))
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

    full_mod_str = full_model_string(term_dict)
    return model_building_utilities.compute(full_mod_str)


def process_likewise_pauli_sum(term):
    r"""
    Terms where the same Pauli is applied to different qubits, summed together.

    Example usage: pauliLikewise_lx_1J2_1J3_2J3_d3
    = XXI + XIX + IXX
    returned as a single matrix

    """
    components = term.split('_')
    components.remove('pauliLikewise')

    connected_sites = []
    for l in components:
        if l[0] == 'd':
            dim = int(l.replace('d', ''))
        elif l[0] == 'l':
            operator = str(l.replace('l', ''))
        else:
            connected_sites.append(l.split('J'))
    all_terms = []
    for s in connected_sites:
        ops = 'J'.join([operator]*len(s))
        conn = 'J'.join(list(s))
        new_term = 'pauliSet_{ops}_{conn}_d{N}'.format(
            ops=ops,
            conn=conn,
            N=dim
        )
        all_terms.append(new_term)

    total_model_string = '+'.join(all_terms)
    return qmla.model_building_utilities.compute(total_model_string)


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
    return model_building_utilities.compute(op_name)


def process_ising_chain(term):
    print("USING ISING CHAIN MTX PROCESS for ", term)
    components = term.split('_')
    components.remove('isingChain')

    for element in components:
        if element[0] == 'd':
            dim = int(element.replace('d', ''))

    mtx = None
    for d in range(1, dim):
        s = 'pauliSet_{}J{}_zJz_d{}'.format(d, d + 1, dim)
        if mtx is None:
            mtx = qmla.model_building_utilities.compute(s)
        else:
            mtx += qmla.model_building_utilities.compute(s)
    return mtx


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
    running_mtx = model_building_utilities.compute(individual_transverse_terms[0])
    for term in individual_transverse_terms[1:]:
        running_mtx += model_building_utilities.compute(term)
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

    running_mtx = model_building_utilities.compute(individual_interaction_terms[0])

    for term in individual_interaction_terms[1:]:
        running_mtx += model_building_utilities.compute(term)

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

    running_mtx = model_building_utilities.compute(individual_interaction_terms[0])

    for term in individual_interaction_terms[1:]:
        running_mtx += model_building_utilities.compute(term)

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

    running_mtx = model_building_utilities.compute(individual_transverse_terms[0])

    for term in individual_transverse_terms[1:]:
        running_mtx += model_building_utilities.compute(term)

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


def process_fermi_hubbard_onsite_sum(constituents):
    from qmla.process_string_to_matrix import process_basic_operator
    sites = []
    for c in constituents:
        if c[0] == 'd':
            num_sites = int(c[1:])
        else:
            sites.append(int(c))

    mtx = None
    for s in sites:
        onsite_term = "FHonsite_{s}_d{N}".format(
            s=s,
            N=num_sites
        )
        if mtx is None:
            mtx = process_basic_operator(onsite_term)
        else:
            mtx += process_basic_operator(onsite_term)
    return mtx


def process_fermi_hubbard_hopping_sum(constituents):
    from qmla.process_string_to_matrix import process_basic_operator
    connected_sites = []
    for c in constituents:
        if c in ['down', 'up']:
            spin_type = c
        elif c[0] == 'd':
            num_sites = int(c[1:])
        else:
            sites = [int(s) for s in c.split('h')]
            connected_sites.append(sites)

    mtx = None
    for cs in connected_sites:
        hopping_term = "FHhop_{s}_{i}h{k}_d{N}".format(
            s=spin_type,
            i=cs[0],
            k=cs[1],
            N=num_sites
        )
        if mtx is None:
            mtx = process_basic_operator(hopping_term)
        else:
            mtx += process_basic_operator(hopping_term)
    return mtx


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
        if c == 'FH-hopping-sum':
            constituents.remove(c)
            mtx = process_fermi_hubbard_hopping_sum(constituents)
        elif c == 'FHhop':
            constituents.remove(c)
            mtx = process_fermi_hubbard_hopping(constituents)
        elif c == 'FH-onsite-sum':
            constituents.remove(c)
            mtx = process_fermi_hubbard_onsite_sum(constituents)
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
    # for use in computing base level terms in a model, used in
    # model_building_utilities.
    return base_hubbard_grouped_term(term)
