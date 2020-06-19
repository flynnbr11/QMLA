from __future__ import print_function

import numpy as np
import copy
import pandas as pd

import qmla.logging

__all__ = [
    'Operator',
    'core_operator_dict',
    'get_num_qubits',
    'get_constituent_names_from_name',
    'alph',
    'consider_new_model',
    'reduced_model_instance_from_id',
    'update_field',
    'pull_field',
    'check_model_exists',
    'unique_model_pair_identifier',
    'all_active_model_ids',
    'model_id_from_name',
    'list_model_id_in_branch'
]

core_operator_dict = {
    'i': np.array([[1 + 0.j, 0 + 0.j], [0 + 0.j, 1 + 0.j]]),
    'x': np.array([[0 + 0.j, 1 + 0.j], [1 + 0.j, 0 + 0.j]]),
    'y': np.array([[0 + 0.j, 0 - 1.j], [0 + 1.j, 0 + 0.j]]),
    'z': np.array([[1 + 0.j, 0 + 0.j], [0 + 0.j, -1 + 0.j]]),
    'a': np.array([[0 + 0.j, 1 + 0.j], [0 + 0.j, 0 + 0.j]]),
    's': np.array([[0 + 0.j, 0 + 0.j], [1 + 0.j, 0 + 0.j]])
}



"""
------ ------ Operator Class ------ ------
"""


class Operator():
    r"""
    Operator objects for Hamiltonian models.

    Translates a model name (string) into:
    - constituents_names: strings specifying constituents
    - constituents_operators: whole matrices of constituents
    - num_qubits: total dimension of operator [number of qubits it acts on]
    - matrix: total matrix operator
    - qubits_acted_on: list of qubits which are acted on non-trivially
      -- e.g. xTiTTz has list [1,3], since qubit 2 is acted on by identity
    - alph_name: rearranged version of name which follows alphabetical convention
      -- uniquely identifies equivalent operators for comparison
            against previously considered models

    :param str name: name of model

    """

    def __init__(self, name):
        self.name = name

    @property
    def constituents_names(self):
        """
        List of constituent operators names.
        """

        return get_constituent_names_from_name(self.name)

    @property
    def num_qubits(self):
        """
        Number of qubits this operator acts on.
        """
        return get_num_qubits(self.name)

    @property
    def constituents_operators(self):
        """
        List of matrices of constituents.
        """
        ops = []
        for i in self.constituents_names:
            ops.append(compute(i))
        return ops

    @property
    def num_constituents(self):
        """
        Integer, how many constituents, and therefore parameters, are in this model.
        """
        return len(self.constituents_names)

    @property
    def matrix(self):
        """
        Full matrix of operator.
        """
        mtx = None
        for i in self.constituents_operators:
            if mtx is None:
                mtx = i
            else:
                mtx += i
        return mtx

    @property
    def alph_name(self):
        """
        Name of operator rearranged to conform with alphabetical naming convention.
        Uniquely identifies equivalent operators.
        For use when comparing potential new operators.
        """
        return alph(self.name)

    @property
    def eigenvectors(self):
        return get_eigenvectors(self.name)


"""
Functions for use by operator class to parse string (name) and prodcue relevent operators, lists etc.
"""
def get_num_qubits(name):
    """
    Parse string and determine number of qubits this operator acts on.
    """
    t_str, p_str, max_t, max_p = get_t_p_strings(name)
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

    # if hopping term wasn't found in individual terms
    max_t_found = 0
    t_str = ''
    while name.count(t_str + 'T') > 0:
        t_str = t_str + 'T'
    num_qubits = len(t_str) + 1

    return num_qubits


def get_constituent_names_from_name(name):
    return name.split('+')

def DEPRECATED_get_constituent_names_from_name(name):
    r""" 
    Used when terms are separated by P strings, such as xTiPPyTi, 
    i.e. the old method. Now all terms should be separable by +
    (e.g. xTi+yTi)
    """
    verbose_naming_mechanism_terms = ['T', 'P', 'M']

    if np.any(
        [t in name for t in verbose_naming_mechanism_terms]
    ):
        return verbose_naming_mechanism_separate_terms(name)
    else:
        return name.split('+')


def verbose_naming_mechanism_separate_terms(name):
    t_str, p_str, max_t, max_p = get_t_p_strings(name)
    if(max_t >= max_p):
        # if more T's than P's in name,
        # it has only one constituent.
        return [name]
    else:
        # More P's indicates a sum at the highest dimension.
        return name.split(p_str)


def get_t_p_strings(name):
    """
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


def find_max_letter(string, letter):
    """
    Find largest instance of consecutive given 'letter'.
    Return largest instance and length of that instance.
    """
    letter_str = ''
    while string.count(letter_str + letter) > 0:
        letter_str = letter_str + letter

    return len(letter_str), letter_str


def empty_array_of_same_dim(name):
    """
    Parse name to find size of system it acts on.
    Produce an empty matrix of that dimension and return it.
    """
    num_qubits = get_num_qubits(name)
    dim = 2**num_qubits
    empty_mtx = np.zeros([dim, dim], dtype=np.complex128)
    return empty_mtx


def alph(name):
    """
    Return alphabetised version of name.
    Parse string and recursively call alph function to alphabetise substrings.
    """
    if '+' in name:
        separate_terms = name.split('+')
        alphabetised = '+'.join(sorted(separate_terms))
        return alphabetised

    # TODO rewrite for names separated by +
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


def ideal_probe(name):
    """
    Returns a probe state which is the normalised sum of the given operator's
    eigenvectors, ideal for probing that operator.
    """
    mtx = Operator(name).matrix
    eigvalues = np.linalg.eig(mtx)[1]
    summed_eigvals = np.sum(eigvalues, axis=0)
    normalised_probe = summed_eigvals / np.linalg.norm(summed_eigvals)
    return normalised_probe

def get_eigenvectors(name):
    mtx = Operator(name).matrix
    eigvectors = np.linalg.eig(mtx)[0]
    return eigvectors


"""
------ ------ Functions to interact with database object ------ ------
"""


def get_location(db, name):
    """
    Return which row in db corresponds to the string name.
    """
#    for i in range(len(db['model_name'])):
    for i in list(db.index.values):
        if db['model_name'][i] == name:
            return i


def consider_new_model(model_lists, name, db):
    """
    Check whether the new model, name, exists in all previously considered models,
    held in model_lists.
    If name has not been previously considered, 'New' is returned.
    If name has been previously considered, the corresponding location
        in db is returned.
    TODO: return something else? Called in add_model function.
    Returning 0,1 would cause an error on the condition the function is
        returned into.
    """
    # Return true indicates it has not been considered and so can be added
    al_name = alph(name)
    n_qub = get_num_qubits(name)
    if al_name in model_lists[n_qub]:
        return 'Previously Considered'  # todo -- make clear if in legacy or running db
    else:
        return 'New'


def check_model_in_dict(name, model_dict):
    """
    Check whether the new model, name, exists in all previously considered models,
        held in model_lists.
    If name has not been previously considered, False is returned.
    """
    # Return true indicates it has not been considered and so can be added

    al_name = alph(name)
    n_qub = get_num_qubits(name)

    if al_name in model_dict[n_qub]:
        return True  # todo -- make clear if in legacy or running db
    else:
        return False


def check_model_exists(model_name, model_lists, db):
    # Return True if model exits; False if not.
    if consider_new_model(model_lists, model_name, db) == 'New':
        return False
    else:
        return True


def unique_model_pair_identifier(model_a_id, model_b_id):
    a = int(float(model_a_id))
    b = int(float(model_b_id))
    std = sorted([a, b])
    id_str = ''
    for i in range(len(std)):
        id_str += str(std[i])
        if i != len(std) - 1:
            id_str += ','

    return id_str


def model_id_from_name(db, name):
    name = alph(name)
    return db.loc[db['model_name'] == name]['model_id'].item()


def model_name_from_id(db, model_id):
    print("[DB] model_id:", model_id)
    return db.loc[db['model_id'] == model_id]['model_name'].item()


def index_from_model_id(db, model_id):
    return db.loc[db['model_id'] == model_id].index[0]


def reduced_model_instance_from_id(db, model_id):
    idx = index_from_model_id(db, model_id)
    return db.loc[idx]["Reduced_Model_Class_Instance"]


def list_model_id_in_branch(db, branch_id):
    return list(db[db['branch_id'] == branch_id]['model_id'])


def update_field(db, field, name=None, model_id=None,
                 new_value=None, increment=None):
    if name is not None:
        db.loc[db['model_name'] == name, field] = new_value
    elif model_id is not None:
        db.loc[db['model_id'] == model_id, field] = new_value


def pull_field(db, name, field):
    idx = get_location(db, name)
    if idx is not None:
        return db.loc[idx, field]
    else:
        print("Cannot update field -- model does not exist in database_framework.")


def all_active_model_ids(db):
    return list(db[(db['Status'] == 'Active')]['model_id'])


def active_model_ids_by_branch_id(db, branch_id):
    return list(db[(db['branch_id'] == branch_id) & (
        db['Status'] == 'Active')]['model_id'])
