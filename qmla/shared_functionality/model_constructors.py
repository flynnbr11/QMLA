from __future__ import print_function

import numpy as np
import copy
import pandas as pd
import sys

from qmla.model_building_utilities import (
    core_operator_dict,
    get_num_qubits,
    alph,
    get_constituent_names_from_name,
    unique_model_pair_identifier,
    compute,
)
from qmla.shared_functionality import latex_model_names
from qmla.process_string_to_matrix import process_basic_operator
from qmla.string_processing_functions import (
    process_multipauli_term,
    process_likewise_pauli_sum,
    process_fermi_hubbard_term,
)
import qmla.logging

##########
# Section: BaseModel object
##########


def log_print(to_print_list, log_file="qmd.log", log_identifier="Model Constructor"):
    r"""Writng to unique QMLA instance log."""
    qmla.logging.print_to_log(
        to_print_list=to_print_list, log_file=log_file, log_identifier=log_identifier
    )


class BaseModel:
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

    def __init__(self, name, fixed_parameters=None, **kwargs):
        self.name = alph(name)
        self.fixed_parameters = fixed_parameters
        print("BASE MODEL fixed_parameters : ", fixed_parameters)

        # Modular functionality
        self.latex_name_method = (
            qmla.shared_functionality.latex_model_names.pauli_set_latex_name
        )
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
            self.model_specific_basic_operator(term) for term in self.terms_names
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
            return self.construct_matrix(copy.deepcopy(self.fixed_parameters))
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
        mtx = np.tensordot(np.array(parameters), np.array(self.terms_matrices), axes=1)
        return mtx

    def model_specific_basic_operator(self, term):
        # process a basic term in the formalism of this model
        # this can use a prebuilt fnc, or build one from scratch without relying on compute() etc.
        # if using a prebuilt, set self.basic_string_processer

        return self.basic_string_processer(term)

    @property
    def model_prior(self):
        # TODO move prior definition here and all calls to get_prior go via exploration_strategy
        # to pick up means/widths, then through here

        return


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
        return 2 * get_num_qubits(self.name)


class SharedParametersModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def construct_matrix(self, parameters):
        r"""
        combine terms in matrix, e.g.
        a * (t0 + t1) + b *t2
        """
        print("Constucting model. parameters={}".format(parameters))
        print("params have type {}".format(type(parameters)))
        try:
            mtx = parameters[0] * (self.terms_matrices[0] + self.terms_matrices[1])
            mtx += parameters[1] * self.terms_matrices[2]
        except:
            print(
                "can't construct mtx from parameters = {} and terms \n {}".format(
                    parameters, self.terms_matrices
                )
            )
            raise
        return mtx

    @property
    def num_parameters(self):
        return len(self.parameters_names) - 1

    @property
    def terms_names_latex(self):
        return [r"$\alpha$", r"$\beta$"]

    def model_prior(self):
        # TODO restructure where prior gets called, so it produces fewer parameters here than the name
        return


class LiouvillianModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latex_name_method = self.latex_name_method_custom

    def liouvillian_parameter_names(self):
        string = self.name
        initial_strings = string.split("+")
        output = []
        for term in initial_strings:

            if "HamLiouvillian" in term:
                output.append(term)
            elif "DissLiouvillian" in term:
                terms = "_".join(term.split("_")[1:-1])
                diss_terms = terms.split("_~_")
                for i in diss_terms:
                    output.append("DissLiouvillian_" + i)
            else:
                raise NameError(term)
        return output

    def latex_name_method_custom(self, name, **kwargs):
        name = name.replace("_", "")
        return "$" + name + "$"

    @property
    def terms_names(self):
        # in general may not be the same
        return self.liouvillian_parameter_names()

    def lindblad_to_liouvillian(self, mtx):
        """
        Taking in a single dissipator matrix, this function converts the matrix to one operating
        in Liouvillian space.
        """
        term_1 = np.kron(mtx, mtx.conj())

        term_2 = -0.5 * (
            np.kron(np.dot(mtx.conj().T, mtx), np.eye(int(np.sqrt(mtx.size))))
        )

        term_3 = -0.5 * (
            np.kron(np.eye(int(np.sqrt(mtx.size))), np.dot(mtx.conj().T, mtx).T)
        )

        return term_1 + term_2 + term_3

    def hamiltonian_to_liouvillian(self, mtx):
        """
        Taking in a matrix, which should be a matrix moves it to Liouvillian space, preforming the transformation through the
        Von Neumann Equation.
        """
        return -1j * (
            np.kron(mtx, np.eye(int(np.sqrt(mtx.size))))
            - np.kron(np.eye(int(np.sqrt(mtx.size))), mtx.T)
        )

    def list_to_mtx(self, letter_list):
        """
        Taking in a list of lists of single letter strings this function returns a matrix by looking up
        a dictionary in qmla.

        To construct x*I*I+I*x*I+I*I*x where * denotes a tensor product use : [['x','i','i'],['i','x','i'],['i','i','x']]
        """
        op = np.zeros(
            (2 ** len(letter_list[0]), 2 ** len(letter_list[0])), dtype=np.complex128
        )
        for x in letter_list:
            mini_op = 1
            for y in x:
                mini_op = np.kron(
                    mini_op, qmla.model_building_utilities.core_operator_dict[y]
                )
            op = op + mini_op
        return op

    def construct_matrix(self, parameters, string=None):
        """
        Given a string of a Liouvillian system and a list of parameters this function will return
        the full matrix, provided the parameters are ordered alphabetically.
        """

        param_list = parameters

        if type(param_list) == type(np.array([0])):
            param_list = param_list.tolist()

        if string == None:
            string = self.name

        if "Liouvillian" in string:
            # Liouvillian Handeling
            string_terms = string.split("+")
            qubits = float(string_terms[0].split("_")[-1][1:])
            output_mtx = np.zeros(
                (int(4 ** qubits), int(4 ** qubits)), dtype=np.complex128
            )
            for i in string_terms:
                if i.split("_")[0] == "DissLiouvillian":
                    # Dissipator Handeling
                    size = int(i.split("_")[-1][1:])
                    op = np.zeros(
                        (int(2 ** qubits), int(2 ** qubits)), dtype=np.complex128
                    )
                    i = i.split("_")[1:-1]
                    i = ("_".join(i)).split("_~_")
                    for j in i:
                        matrix_type = j.split("_")[1]
                        if matrix_type[0] == "l":
                            matrix_type = matrix_type[1:]

                        position = j.split("_")[2:]
                        empty = [["i"] * size for x in range(len(position))]

                        for num, x in enumerate(position):
                            if "J" in matrix_type:
                                if "J" not in x:
                                    sys.exit(
                                        "Matrix pointer is the tensor product over two sites but only one site is referenced."
                                    )
                                empty[num][int(x.split("J")[0]) - 1] = matrix_type[0]
                                empty[num][int(x.split("J")[1]) - 1] = matrix_type[-1]
                            elif "J" in x:
                                empty[num][int(x.split("J")[0]) - 1] = matrix_type
                                empty[num][int(x.split("J")[1]) - 1] = matrix_type
                            else:
                                empty[num][int(x) - 1] = matrix_type

                        op = op + self.list_to_mtx(empty) * param_list.pop(0)

                    # i is the collection of matrix decleration withint he dissipation term. params_in_use should be
                    # the parmeters that correspond to those matrix declerations and param_list should be preserved.

                    output_mtx += self.lindblad_to_liouvillian(op)

                elif i.split("_")[0] == "HamLiouvillian":
                    pointer_list = []

                    i = i.split("_")[1:]
                    size = int(i[-1][1:])
                    matrix_type = i[0]
                    position = i[1:-1]
                    # bulk out pointer_list

                    for x in range(len(position)):
                        pointer_list.append([])
                        for y in range(size):
                            pointer_list[x].append("i")

                    for num, x in enumerate(position):
                        if "J" in matrix_type:
                            if "J" not in x:
                                sys.exit(
                                    "Matrix pointer is the tensor product over two sites but only one site is referenced."
                                )
                            pointer_list[num][int(x.split("J")[0]) - 1] = matrix_type[0]
                            pointer_list[num][int(x.split("J")[1]) - 1] = matrix_type[
                                -1
                            ]
                        elif "J" in x:
                            pointer_list[num][int(x.split("J")[0]) - 1] = matrix_type[
                                -1
                            ]
                            pointer_list[num][int(x.split("J")[1]) - 1] = matrix_type[
                                -1
                            ]
                        else:
                            pointer_list[num][int(x) - 1] = matrix_type[-1]
                    op = self.list_to_mtx(pointer_list) * param_list.pop(0)
                    output_mtx += self.hamiltonian_to_liouvillian(op)
                else:
                    sys.exit("Liouvillian term not recognised" + str(i.split("_")[0]))
        else:
            # Non Hamiltonian Modeling
            output_mtx = compute(string)
        string_terms = string.split("+")

        if param_list != []:
            sys.exit(
                "The number of parameters exceeds the number of parameters expected from the terms"
            )

        return output_mtx

    @property
    def num_qubits(self):
        """
        As QHL is normally to work with Hamiltonians the number of qubits can be used, at times, to declare the
        size of a matrix. This will break if one uses the inbuilt method as Liouvillian matrices are much
        larger than Hamiltonians. This function returns the number of qubits that a Hamiltonian would need to
        make the Liouvillian of the size declared in the string.

        The number of qubits in the system, physically is still the number in the dimension component of the string.
        """
        string = self.name
        endings = []
        for i in string.split("+"):
            endings.append(i.split("_")[-1])

        check = all(elem == endings[0] for elem in endings)
        if not check:
            sys.exit("dimension component of system is not consistent")
        endings = int(endings[0][1:]) * 2
        return endings

    @property
    def terms_matrices(self):
        """
        List of matrices of constituents.
        """
        operators = []
        for term in self.name.split("+"):
            operators.append(
                self.construct_matrix(
                    ([1] * self.liouvillian_parameters(string=term)), string=term
                )
            )
        return operators

    @property
    def num_parameters(self):
        return self.liouvillian_parameters()

    def liouvillian_parameters(self, string=None):
        if string == None:
            string = self.name

        num = 0
        for a in string.split("+"):
            if a.split("_")[0] == "HamLiouvillian":
                num += 1
            else:
                for b in a.split("~"):
                    num += 1
        return num

    @property
    def matrix(self):
        mtx = construct_matrix(self, [1] * len(self.terms_names))
        return mtx

    @property
    def terms_names_latex(self):
        return [r"$\alpha$", r"$\beta$"]
