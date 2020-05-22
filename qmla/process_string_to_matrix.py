import numpy as np

from qmla import database_framework
import qmla.string_processing_functions
import qmla.logging

__all__ = [
    'string_processing_functions',
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
    r"""
    Transform a string, representing a term in the model, into a matrix.

    Physical systems have different corresponding string processing functions. 
    There are provided in the dictionary `qmla.process_string_to_matrix.string_processing_functions`.
    There are a number of rules which model strings must obey to be processed properly. 
        * Terms are separated by ``+``.
        * Within terms, components are separated by ``_``.
        * Components have different meanings, depending on which string
          processing function is used. 
        * The first component is the `indicator` of which processing function to use; 
          it is matched with a processing function in `~qmla.string_processing_functions`. 
        * The final component in general indicates the dimension  ``N`` of the system, 
          and is specified by ``dN``. 
        * No other component should start with ``d``, as it uniquely indicates the dimension.
        * Alternatively, core operators can be processed alone, 
          these are given in :attr:`~qmla.core_operator_dict`. 
    For example, the string ``pauliSet_1J2_xJx_d3+pauliSet_1J3_zJz_d3``:
        * Terms: ``pauliSet_1J2_xJx_d3``, ``pauliSet_1J3_zJz_d3``
        * Components: ``pauliSet``, ``1J2``, ``xJx``, ``d3``
        * Indicator ``pauliSet`` tells it to process via :meth:`~qmla.process_multipauli_term`.
        * ``d3`` tells it to use a 3 qubit basis
        * Other components are interpreted by the string processing function
        * In this case, the result is the matrix ( XXI + ZIZ) . 

    :param str basic_operator: term to generate matrix from. 
    :return np.ndarray mtx: matrix corresponding to the input term.
    """

    indicator = basic_operator.split('_')[0]
    if indicator in string_processing_functions:
        mtx = string_processing_functions[indicator](basic_operator)
    else:
        mtx = database_framework.core_operator_dict[basic_operator]

    return mtx

