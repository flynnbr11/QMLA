from __future__ import print_function

import numpy as np
import copy
import pandas as pd

from qmla.model_building_utilities import \
    core_operator_dict, get_num_qubits, alph, \
    get_constituent_names_from_name,  \
    unique_model_pair_identifier, compute
import qmla.logging

__all__ = [
    'BaseModel',
]

##########
# Section: BaseModel object
##########

class BaseModel():
    r"""
    Operator objects for Hamiltonian models.

    Translates a model name (string) into:
        * terms_names: strings specifying constituents
        * terms_matrices: whole matrices of constituents
        * num_qubits: total dimension of operator [number of qubits it acts on]
        * matrix: total matrix operator
        * qubits_acted_on: list of qubits which are acted on non-trivially
        -- e.g. xTiTTz has list [1,3], since qubit 2 is acted on by identity
        * alph_name: rearranged version of name which follows alphabetical convention
        -- uniquely identifies equivalent operators for comparison
                against previously considered models

    :param str name: name of model

    """

    def __init__(
        self, 
        name,
        fixed_parameters=None, 
    ):
        self.name = alph(name)
        self.latex_name_method = qmla.shared_functionality.latex_model_names.pauli_set_latex_name
        self.fixed_parameters = fixed_parameters
            
    @property
    def terms_names(self):
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
    def terms_matrices(self):
        """
        List of matrices of constituents.
        """
        ops = []
        for i in self.terms_names:
            ops.append(compute(i))
        return ops

    @property
    def num_terms(self):
        """
        Integer, number of constituents (and therefore parameters) in this model.
        """
        return len(self.terms_names)

    @property
    def matrix(self):
        """
        Full matrix of operator.
        Assumes weight 1 on each constituent matrix.
        """
        mtx = None
        for i in self.terms_matrices:
            if mtx is None:
                mtx = i
            else:
                mtx += i
        return mtx

    @property
    def name_alphabetical(self):
        """
        Name of operator rearranged to conform with alphabetical naming convention.
        Uniquely identifies equivalent operators.
        For use when comparing potential new operators.
        """
        return alph(self.name)

    @property
    def name_latex(self):
        return self.latex_name_method(self.name)

    @property
    def eigenvectors(self):
        return get_eigenvectors(self.name)

    @property
    def fixed_matrix(self):
        # TODO does this need to be a property?
        if self.fixed_parameters is not None:
            return self.construct_matrix(
                self.fixed_parameters
            )
        else:
            return None

    def construct_matrix(self, parameters):
        r""" 
        Default: 
            sum(p[i] * operators[i])
        """
        mtx = np.tensordot(
            np.array(parameters), 
            np.array(self.terms_matrices), 
            axes=1
        )
        return mtx 

