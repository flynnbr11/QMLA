"""
Functions which find which branch to display models on. 

Called by exploration strategies name_branch_map wrapper function. 
Branch IDs returned are displayed vertically downwards:
    models on branch 0 appear at the top of the DAG; 
    models on branch 1,...,N appear beneath. 
"""

import numpy as np

import qmla.construct_models

def branch_is_num_params(latex_mapping_file, **kwargs):
    r"""
    Number of parameters in the models correspond to branch. 

    :param str latex_mapping_file: path to file containing
        tuples of model strings and their corresponding 
        latex representations, for all models considered in the run. 
    """

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
        num_params = len(qmla.construct_models.get_constituent_names_from_name(mod))
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
        # num_params = len(qmla.construct_models.get_constituent_names_from_name(mod))
        num_qubits = qmla.construct_models.get_num_qubits(mod)
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
        # num_params = len(qmla.construct_models.get_constituent_names_from_name(mod))
        num_qubits = qmla.construct_models.get_num_qubits(mod)
        max_num_params_this_num_sites = 1
        num_params = len(
            qmla.construct_models.get_constituent_names_from_name(mod)
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
        num_qubits = qmla.construct_models.get_num_qubits(mod)
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
            num_params = len(qmla.construct_models.get_constituent_names_from_name(model))
            branch_idx = base_branch + num_params
            if branch_idx > highest_branch:
                highest_branch = branch_idx
            model_branches[model] = branch_idx
    return model_branches

