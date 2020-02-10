import numpy as np

import qmla.database_framework as database_framework

"""
Useful functions
"""


def full_model_string(operations):
    """
    operations must be a dict with elements:
    - 'dim' : number of qubits
    - 'terms' : list of lists of tuple of the form,
        e.g. [ (1, 'x'), (2, 'y')]
        i.e. tuples (qubit_id, pauli_operator)
        Each nested list gives a term, which are all added together for the full model
    Reconstructs unique model name for that Hamiltonian.
    """

    # Note TODO: this doesn't give an error when tuples are
    # given which aren't used. it should
    from qmla.database_framework import alph
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

    full_model = p_str.join(all_terms)
    # full_model = database_framework.alph(full_model)
    full_model = alph(full_model)
    return full_model


def operations_dict_from_name(mod_name):
    constituents = database_framework.get_constituent_names_from_name(mod_name)
    num_qubits = database_framework.get_num_qubits(mod_name)
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

"""
First define all the functions which are callable
by the wrappers for converting name strings to
latex representations
"""


def default_latex_wrapping(name):
    return str('$' + str(name) + '$')


def latex_name_ising(name):
    # TODO generalise this
    # if name == 'zTi': # FOR BQIT19 Poster #TODO REMOVE
    #     return '$\Omega$'

    if name == 'x' or name == 'y' or name == 'z':
        return '$' + name + '$'

    num_qubits = database_framework.get_num_qubits(name)
    terms = name.split('PP')
    rotations = ['xTi', 'yTi', 'zTi']
    hartree_fock = ['xTx', 'yTy', 'zTz']
    transverse = ['xTy', 'xTz', 'yTz', 'yTx', 'zTx', 'zTy']

    present_r = []
    present_hf = []
    present_t = []

    for t in terms:
        if t in rotations:
            present_r.append(t[0])
        elif t in hartree_fock:
            present_hf.append(t[0])
        elif t in transverse:
            string = t[0] + t[-1]
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
        latex_term += 'S_{' + r_terms + '}'
    if len(present_hf) > 0:
        latex_term += 'HF_{' + hf_terms + '}'
    if len(present_t) > 0:
        latex_term += 'T_{' + t_terms + '}'

    final_term = '$' + latex_term + '$'
    if final_term != '$$':
        return final_term

    else:
        plus_string = ''
        for i in range(num_qubits):
            plus_string += 'P'
        individual_terms = name.split(plus_string)
        individual_terms = sorted(individual_terms)

        latex_term = '+'.join(individual_terms)
        final_term = '$' + latex_term + '$'
        return final_term


############# Latex naming functions used by growth rules ##########
 
def large_spin_bath_nv_system_name(term):
    num_qubits = database_framework.get_num_qubits(term)
    t_str = 'T' * (num_qubits - 1)
    p_str = 'P' * num_qubits
    separate_terms = term.split(p_str)

    spin_terms = []
    interaction_terms = []

    for t in separate_terms:
        components = t.split('_')
        components.remove('nv')
        components.remove(str('d' + str(num_qubits)))
        if 'spin' in components:
            components.remove('spin')
            spin_terms.append(components[0])
        elif 'interaction' in components:
            components.remove('interaction')
            interaction_terms.append(components[0])

    latex_name = '('
    if len(spin_terms) > 0:
        latex_name += 'S_{'
        for s in spin_terms:
            latex_name += str(s)
        latex_name += '}'
    if len(interaction_terms) > 0:
        latex_name += 'I_{'
        for s in interaction_terms:
            latex_name += str(s)
        latex_name += '}'

    latex_name += str(
        r')^{\otimes'
        + str(num_qubits)
        + '}'
    )

    return '$' + latex_name + '$'


def pauliSet_latex_name(
    name,
    **kwargs
):
    core_operators = list(sorted(database_framework.core_operator_dict.keys()))
    num_sites = database_framework.get_num_qubits(name)
    p_str = 'P' * num_sites
    separate_terms = name.split(p_str)

    latex_terms = []
    term_type_markers = ['pauliSet', 'transverse']
    for term in separate_terms:
        components = term.split('_')
        if 'pauliSet' in components:
            components.remove('pauliSet')

            for l in components:
                if l[0] == 'd':
                    dim = int(l.replace('d', ''))
                elif l[0] in core_operators:
                    operators = l.split('J')
                else:
                    sites = l.split('J')

            latex_str = r'\sigma'

            latex_str += '^{'
            for s in sites:
                latex_str += str('{},'.format(s))

            latex_str = latex_str[0:-1]
            latex_str += '}'

            latex_str += '_{'
            for o in operators:
                latex_str += str('{},'.format(o))
            latex_str = latex_str[0:-1]  # remove final comma
            latex_str += '}'

        elif 'transverse' in components:
            components.remove('transverse')
            for l in components:
                if l[0] == 'd':
                    dim = int(l.replace('d', ''))
                else:
                    transverse_axis = str(l)

            latex_str = r'\sigma'

            latex_str += r'^{\otimes '
            latex_str += str(dim)
            latex_str += '}'

            latex_str += '_{'
            latex_str += str(transverse_axis)
            latex_str += '}'

        latex_terms.append(latex_str)

    latex_terms = sorted(latex_terms)
    full_latex_term = ''.join(latex_terms)
    full_latex_term = str('$' + full_latex_term + '$')
    full_latex_term = str("'{}'".format(full_latex_term))
    # print("LATEX NAME:", full_latex_term)
    return full_latex_term


############# Model to Branch convention functions ##########

def branch_is_num_params(latex_mapping_file, **kwargs):
    with open(latex_mapping_file) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    latex_name_map = {}
    for c in content:
        this_tuple = eval(c)
        model_string = this_tuple[0]
        latex_name = this_tuple[1]
        # this mapping assigns models to branches with the number of parameters
        # they have
        latex_name_map[model_string] = latex_name

    model_names = list(set(list(latex_name_map.keys())))
    num_params_by_mod = {}
    model_branches = {}

    for mod in model_names:
        num_params = len(database_framework.get_constituent_names_from_name(mod))
        num_params_by_mod[mod] = num_params
        latex_name = latex_name_map[mod]
        model_branches[latex_name] = num_params

    return model_branches


def branch_is_num_dims(latex_mapping_file, **kwargs):
    with open(latex_mapping_file) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    latex_name_map = {}
    for c in content:
        this_tuple = eval(c)
        model_string = this_tuple[0]
        latex_name = this_tuple[1]
        # this mapping assigns models to branches with the number of parameters
        # they have
        latex_name_map[model_string] = latex_name

    model_names = list(set(list(latex_name_map.keys())))
    model_branches = {}

    for mod in model_names:
        # num_params = len(database_framework.get_constituent_names_from_name(mod))
        num_qubits = database_framework.get_num_qubits(mod)
        latex_name = latex_name_map[mod]
        model_branches[latex_name] = num_qubits

    return model_branches


def branch_is_num_params_and_qubits(
    latex_mapping_file,
    **kwargs
):
    with open(latex_mapping_file) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    latex_name_map = {}
    for c in content:
        this_tuple = eval(c)
        model_string = this_tuple[0]
        latex_name = this_tuple[1]
        # this mapping assigns models to branches with the number of parameters
        # they have
        latex_name_map[model_string] = latex_name

    model_names = list(set(list(latex_name_map.keys())))
    model_branches = {}

    for mod in model_names:
        # num_params = len(database_framework.get_constituent_names_from_name(mod))
        num_qubits = database_framework.get_num_qubits(mod)
        max_num_params_this_num_sites = 1
        num_params = len(
            database_framework.get_constituent_names_from_name(mod)
        )
        latex_name = latex_name_map[mod]
        branch_num = (max_num_params_this_num_sites * num_qubits) + num_params
        model_branches[latex_name] = branch_num

    return model_branches


def branch_computed_from_qubit_and_param_count(
    latex_mapping_file,
    **kwargs
):
    with open(latex_mapping_file) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    latex_name_map = {}
    for c in content:
        this_tuple = eval(c)
        model_string = this_tuple[0]
        latex_name = this_tuple[1]
        # this mapping assigns models to branches with the number of parameters
        # they have
        latex_name_map[model_string] = latex_name

    model_names = list(set(list(latex_name_map.keys())))

    models_by_num_qubits = {}
    for mod in model_names:
        num_qubits = database_framework.get_num_qubits(mod)
        if num_qubits not in models_by_num_qubits.keys():
            models_by_num_qubits[num_qubits] = []
        models_by_num_qubits[num_qubits].append(mod)

    numbers_of_qubits = sorted(models_by_num_qubits.keys())
    model_branches = {}
    highest_branch = 0

    for num_qubits in numbers_of_qubits:
        base_branch = highest_branch
        models = models_by_num_qubits[num_qubits]
        for model in models:
            num_params = len(database_framework.get_constituent_names_from_name(model))
            branch_idx = base_branch + num_params
            if branch_idx > highest_branch:
                highest_branch = branch_idx
            model_branches[model] = branch_idx
    return model_branches

