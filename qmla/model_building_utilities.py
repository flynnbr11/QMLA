from __future__ import print_function

import numpy as np
import copy
import pandas as pd

import qmla.logging

__all__ = [
    'core_operator_dict',
    'get_num_qubits',
    'get_constituent_names_from_name',
    'alph',
    'unique_model_pair_identifier',
]

##########
# Section: Core operators as arrays
##########

core_operator_dict = {
    'i': np.array([  # Identity
        [1 + 0.j, 0 + 0.j],
        [0 + 0.j, 1 + 0.j]
    ]),
    'x': np.array([  # Pauli-X
        [0 + 0.j, 1 + 0.j],
        [1 + 0.j, 0 + 0.j]
    ]),
    'y': np.array([  # Pauli-Y
        [0 + 0.j, 0 - 1.j],
        [0 + 1.j, 0 + 0.j]
    ]),
    'z': np.array([  # Pauli-Z
        [1 + 0.j, 0 + 0.j],
        [0 + 0.j, -1 + 0.j]
    ]),
    'a': np.array([  # Add
        [0 + 0.j, 1 + 0.j],
        [0 + 0.j, 0 + 0.j]
    ]),
    's': np.array([  # Subtract
        [0 + 0.j, 0 + 0.j],
        [1 + 0.j, 0 + 0.j]
    ])
}

##########
# Section: functions for constructing models.
# compte methods are called recursively on names to
# construct matrices corresponding to input model names
##########


def compute_t(inp):
    """
    Assuming largest instance of action on inp is tensor product, T.
    Parse string.
    Recursively call compute() function.
    Tensor product resulting lists.
    Return operator which is specified by inp.
    """
    max_t, t_str = find_max_letter(inp, "T")
    max_p, p_str = find_max_letter(inp, "P")

    if(max_p == 0 and max_t == 0):
        pauli_symbol = inp
        return core_operator_dict[pauli_symbol]

    elif(max_t == 0):
        return compute(inp)
    else:
        to_tens = inp.split(t_str)
        running_tens_prod = compute(to_tens[0])
        for i in range(1, len(to_tens)):
            max_p, p_str = find_max_letter(to_tens[i], "P")
            max_t, t_str = find_max_letter(to_tens[i], "T")
            rhs = compute(to_tens[i])
            running_tens_prod = np.kron(running_tens_prod, rhs)
        return running_tens_prod


def compute_p(inp):
    """
    Assuming largest instance of action on inp is addition, P.
    Parse string.
    Recursively call compute() function.
    Sum resulting lists.
    Return operator which is specified by inp.
    """
    max_p, p_str = find_max_letter(inp, "P")
    max_t, t_str = find_max_letter(inp, "T")

    if '+' in inp:
        p_str = '+'
    elif(max_p == 0 and max_t == 0):
        pauli_symbol = inp
        return core_operator_dict[pauli_symbol]
    elif max_p == 0:
        return compute(inp)
    to_add = inp.split(p_str)
    running_sum = empty_array_of_same_dim(to_add[0])
    for i in range(len(to_add)):
        max_p, p_str = find_max_letter(to_add[i], "P")
        max_t, t_str = find_max_letter(to_add[i], "T")
        rhs = compute(to_add[i])
        running_sum += rhs

    return running_sum


def compute_m(inp):
    """
    Assuming largest instance of action on inp is multiplication, M.
    Parse string.
    Recursively call compute() function.
    Multiple resulting lists.
    Return operator which is specified by inp.
    """

    max_m, m_str = find_max_letter(inp, "M")
    max_p, p_str = find_max_letter(inp, "P")
    max_t, t_str = find_max_letter(inp, "T")

    if(max_m == 0 and max_t == 0 and max_p == 0):
        pauli_symbol = inp
        return core_operator_dict[pauli_symbol]

    elif max_m == 0:
        return compute(inp)

    else:
        to_mult = inp.split(m_str)
        #print("To mult : ", to_mult)
        t_str = ''
        while inp.count(t_str + 'T') > 0:
            t_str = t_str + 'T'

        num_qubits = len(t_str) + 1
        dim = 2**num_qubits

        running_product = np.eye(dim)

        for i in range(len(to_mult)):
            running_product = np.dot(running_product, compute(to_mult[i]))

        return running_product


def compute(inp):
    """
    Parse string.
    Recursively call compute() functions (compute_t, compute_p, compute_m).
    Tensor product, multiply or sum resulting lists.
    Return operator which is specified by inp.
    """
    from qmla.process_string_to_matrix import process_basic_operator
    max_p, p_str = find_max_letter(inp, "P")
    max_t, t_str = find_max_letter(inp, "T")
    max_m, m_str = find_max_letter(inp, "M")

    if '+' in inp:
        return compute_p(inp)
    if (max_m == 0 and max_t == 0 and max_p == 0):
        basic_operator = inp
        # call subroutine which can interpret a "basic operator"
        # basic operators are defined with the function
        # they are terms which can not be separated further by P,M,T or +
        return process_basic_operator(basic_operator)
    elif max_m > max_t:
        return compute_m(inp)
    elif max_t >= max_p:
        return compute_t(inp)
    else:
        return compute_p(inp)


##########
# Section: functions for dissecting models
##########

def alph(name):
    r"""
    Alphabetise the model name.

    If name newer follows convention where terms are separated by +, simply separate them.
    If name follows older convention, analyse to separate terms and then alphabetise them.
    Parse string and recursively call alph function to alphabetise substrings.

    :param str name: name of model to alphabetise
    """

    if '+' in name:
        separate_terms = name.split('+')
        alphabetised = '+'.join(sorted(separate_terms))
        return alphabetised

    t_max, t_str = find_max_letter(name, "T")
    p_max, p_str = find_max_letter(name, "P")
    m_max, m_str = find_max_letter(name, "M")

    if p_max == 0 and t_max == 0 and p_max == 0:
        return name

    if p_max > t_max and p_max > m_max:
        ltr = 'P'
        string = p_str
    elif t_max >= p_max:
        string = t_str
        ltr = 'T'
    elif m_max >= p_max:
        string = m_str
        ltr = 'M'
    elif t_max > m_max:
        string = t_str
        ltr = 'T'
    else:
        ltr = 'M'
        string = m_str

    spread = name.split(string)
    if p_max == m_max and p_max > t_max:
        string = p_str
        list_elements = name.split(p_str)

        for i in range(len(list_elements)):
            list_elements[i] = alph(list_elements[i])
        sorted_list = sorted(list_elements)
        linked_sorted_list = p_str.join(sorted_list)
        return linked_sorted_list

    if ltr == 'P' and p_max == 1:
        sorted_spread = sorted(spread)
        out = string.join(sorted_spread)
        return out
    elif ltr == 'P' and p_max > 1:
        list_elements = name.split(string)
        sorted_list = sorted(list_elements)
        for i in range(len(sorted_list)):
            sorted_list[i] = alph(sorted_list[i])
        linked_sorted_list = string.join(sorted_list)
        return linked_sorted_list
    else:
        for i in range(len(spread)):
            spread[i] = alph(spread[i])
        out = string.join(spread)
        return out


def get_num_qubits(name):
    r"""
    Parse string and determine number of qubits this operator acts on.

    Default convention is to use a naming mechanism specified by
    :func:`~qmla.string_processing_functions`.
    In all such constructions, the final element of each term is `dN`,
    so we can extract the number of qubits N.

    If using old convention where terms are tensor-producted
    by T, TT, TTT... ,
    we find the largest T string instance, from which we deduce the number of qubits.
    - xTx = pauli_x TENSOR_PRODUCT pauli_x --> 2 qubits
    - yTyTTy = pauliy_y TENSOR_PRODUCT pauli_y TENSOR_PRODUCT pauli_y --> N=3
    i.e. the largest tensor product of of length is N-1.

    :param str name: name of model
    """

    individual_terms = get_constituent_names_from_name(name)
    for term in individual_terms:
        if (
            term[0:1] == 'h_'
            or '1Dising' in term
            or 'Heis' in term
            or 'nv' in term
            or 'pauliSet' in term
            or 'transverse' in term
            or 'FH' in term
            or 'pauliLikewise' in term
        ):
            terms = term.split('_')
            dim_term = terms[-1]
            dim = int(dim_term[1:])
            num_qubits = dim
            return num_qubits

    max_t_found = 0
    t_str = ''
    while name.count(t_str + 'T') > 0:
        t_str = t_str + 'T'
    num_qubits = len(t_str) + 1

    return num_qubits


def get_constituent_names_from_name(name):
    r"""
    Separate into separate terms in model name.
    e.g. 'pauliSet_1_x_d2+pauliSet_1_y_d2'
    -> ['pauliSet_1_x_d2', 'pauliSet_1_y_d2']
    :param str name: name of model
    """
    return name.split('+')


def empty_array_of_same_dim(name):
    """
    Parse name to find size of system it acts on.
    Produce an empty matrix of that dimension and return it.
    """
    num_qubits = get_num_qubits(name)
    dim = 2**num_qubits
    empty_mtx = np.zeros([dim, dim], dtype=np.complex128)
    return empty_mtx


def find_max_letter(string, letter):
    r"""
    Find largest instance of consecutive given 'letter'.
    Return largest instance and length of that instance.
    """
    letter_str = ''
    while string.count(letter_str + letter) > 0:
        letter_str = letter_str + letter

    return len(letter_str), letter_str


def ideal_probe(name):
    """
    Returns a probe state which is the normalised sum of the given operator's
    eigenvectors, ideal for probing that operator.
    """
    mtx = BaseModel(name).matrix
    eigvalues = np.linalg.eig(mtx)[1]
    summed_eigvals = np.sum(eigvalues, axis=0)
    normalised_probe = summed_eigvals / np.linalg.norm(summed_eigvals)
    return normalised_probe


def get_eigenvectors(name):
    r"""
    Get eigenvectors of a model from its name.
    """

    mtx = BaseModel(name).matrix
    eigvectors = np.linalg.eig(mtx)[0]
    return eigvectors


def unique_model_pair_identifier(model_a_id, model_b_id):
    r"""
    Pair uniquely coupling to model ids, for consistency.
    Formatted as 'low_id,high_id',
    """

    a = int(float(model_a_id))
    b = int(float(model_b_id))
    std = sorted([a, b])
    id_str = ''
    for i in range(len(std)):
        id_str += str(std[i])
        if i != len(std) - 1:
            id_str += ','

    return id_str

##########
# Section: deprecated functions, to be removed when safe to do so
##########

def verbose_naming_mechanism_separate_terms(name):
    r"""
    Separate terms of a model name according to old "verbose" naming scheme.
    """

    t_str, p_str, max_t, max_p = get_t_p_strings(name)
    if(max_t >= max_p):
        # if more T's than P's in name,
        # it has only one constituent.
        return [name]
    else:
        # More P's indicates a sum at the highest dimension.
        return name.split(p_str)


def get_t_p_strings(name):
    r"""
    Find largest instance of consecutive P's and T's.
    Return those instances and lengths of those instances.
    """
    t_str = ''
    p_str = ''
    while name.count(t_str + 'T') > 0:
        t_str = t_str + 'T'

    while name.count(p_str + 'P') > 0:
        p_str = p_str + 'P'

    max_t = len(t_str)
    max_p = len(p_str)

    return t_str, p_str, max_t, max_p
