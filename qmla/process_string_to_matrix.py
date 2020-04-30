import numpy as np

from fermilib.ops import FermionOperator
import fermilib.transforms

from qmla import database_framework
import qmla.string_processing_functions
import qmla.logging

__all__ = [
    'process_basic_operator'
]

string_processing_functions = {
    'nv' : qmla.string_processing_functions.process_n_qubit_NV_centre_spin, 
    'pauliSet' : qmla.string_processing_functions.process_multipauli_term, 
    'pauliLikewise' : qmla.string_processing_functions.process_likewise_pauli_sum,
    'FHhop' : qmla.string_processing_functions.process_fermi_hubbard_term,
    'FHonsite' : qmla.string_processing_functions.process_fermi_hubbard_term,
    'FHchemical' : qmla.string_processing_functions.process_fermi_hubbard_term,
    'FH-hopping-sum' : qmla.string_processing_functions.process_fermi_hubbard_term,
    'FH-onsite-sum' : qmla.string_processing_functions.process_fermi_hubbard_term,
}

def process_basic_operator(basic_operator):
    indicator = basic_operator.split('_')[0]
    if indicator in string_processing_functions:
        mtx = string_processing_functions[indicator](basic_operator)
    else:
        mtx = database_framework.core_operator_dict[basic_operator]

    return mtx

