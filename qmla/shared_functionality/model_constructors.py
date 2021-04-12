from __future__ import print_function

import numpy as np
import copy
import pandas as pd

from qmla.model_building_utilities import \
    core_operator_dict, get_num_qubits, alph, \
    get_constituent_names_from_name,  \
    unique_model_pair_identifier, compute
from qmla.shared_functionality import latex_model_names
from qmla.process_string_to_matrix import process_basic_operator
from qmla.string_processing_functions import \
    process_multipauli_term, process_likewise_pauli_sum, process_fermi_hubbard_term
import qmla.logging

##########
# Section: BaseModel object
##########

class BaseModel():
    r"""
    BaseModel objects for Hamiltonian models.

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
        **kwargs
    ):
        self.name = alph(name)
        self.fixed_parameters = fixed_parameters
        print("BASE MODEL fixed_parameters : ", fixed_parameters)

        # Modular functionality
        self.latex_name_method = qmla.shared_functionality.latex_model_names.pauli_set_latex_name
        self.basic_string_processer = process_multipauli_term        
            
    @property
    def terms_names(self):
        """
        List of constituent operators names.
        """

        return get_constituent_names_from_name(self.name)
    
    @property
    def terms_names_latex(self):
        return [self.latex_name_method(t) for t in self.terms_names]

    @property
    def parameters_names(self):
        # in general may not be the same
        return self.terms_names

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

        operators = [
            self.model_specific_basic_operator(term)
            for term in self.terms_names
        ]
        return operators
    
    @property
    def num_parameters(self):
        return len(self.parameters_names)

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
        Enables custom logic to compute a matrix based on a list of parameters.

        For instance, QInfer-generated particles are passed as an ordered list
        for use within the likelihood function. 
    
        Default: 
            sum(p[i] * operators[i])
        """
        mtx = np.tensordot(
            np.array(parameters), 
            np.array(self.terms_matrices), 
            axes=1
        )
        return mtx 

    def model_specific_basic_operator(self, term):
        # process a basic term in the formalism of this model
        # this can use a prebuilt fnc, or build one from scratch without relying on compute() etc. 
        # if using a prebuilt, set self.basic_string_processer
        
        return self.basic_string_processer(term)

class PauliLikewiseModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.basic_string_processer = process_likewise_pauli_sum
        self.latex_name_method = latex_model_names.lattice_set_grouped_pauli
    
class FermilibModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.basic_string_processer = process_fermi_hubbard_term
        self.latex_name_method = latex_model_names.lattice_set_fermi_hubbard
    
    @property
    def num_qubits(self):
        """
        Number of qubits this operator acts on.
        """
        return 2*get_num_qubits(self.name)


class SharedParametersModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



        