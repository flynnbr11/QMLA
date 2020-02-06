import numpy as np
import pandas as pd

import qmla.model_instances
import qmla.database_framework
import qmla.logging

__all__ = [
    'launch_db',
    'add_model'
]

def log_print(to_print_list, log_file):
    qmla.logging.print_to_log(
        to_print_list = to_print_list,
        log_file = log_file, 
        log_identifier = 'Database launch'
    )



def launch_db(
    new_model_branches,
    new_model_ids,
    log_file,
    true_model_terms_matrices=[],
    true_model_terms_params=[],
    qid=0,
    host_name='localhost',
    port_number=6379,
    **kwargs
):
    """
    Inputs:
    TODO

    Outputs:
      - db: "running database", info on active models. Can access QML and
        operator instances for all information about models.
      - legacy_db: when a model is terminated, we maintain essential information
        in this db (for plotting, debugging etc.).
      - model_lists = list of lists containing alphabetised model names.
        When a new model is considered, it

    Usage:
        $ gen_list = ['xTy, yPz, iTxTTy] # Sample list of model names
        $ running_db, legacy_db, model_lists = database_framework.launch_db(gen_list=gen_list)

    """

    model_lists = { 
        # assumes maxmium 13 qubit-models considered
        # to be checked when checking model_lists
        # TODO generalise or add dimension if not present
        j : []
        for j in range(1,13)
    }
    db = pd.DataFrame({
        '<Name>': [],
        'Status': [],  
        'Completed': [], 
        'branch_id': [],  
        'Reduced_Model_Class_Instance': [],
        'Operator_Instance': [],
        'Epoch_Start': [],
        'ModelID': [],
    })

    model_id = int(0)
    for model_name in list(new_model_branches.keys()):
        try_add_model = add_model(
            model_name=model_name,
            model_id=int(new_model_ids[model_name]),
            branch_id=new_model_branches[model_name],
            running_database=db,
            model_lists=model_lists,
            true_model_terms_matrices=true_model_terms_matrices,
            true_model_terms_params=true_model_terms_params,
            log_file=log_file,
            qid=qid,
            host_name=host_name,
            port_number=port_number
        )
        if try_add_model is True:
            model_id += int(1)

    return db, model_lists



def add_model(
    model_name,
    model_id,
    branch_id,
    running_database,
    model_lists,
    true_model_terms_matrices,
    true_model_terms_params,
    log_file,
    qid,
    host_name='localhost',
    port_number=6379,
    force_create_model=False,
    **kwargs
):
    """
    Function to add a model to the existing databases.
    First checks whether the model already exists.
    If so, does not add model to databases.
      TODO: do we want to return False in this case and use as a check in QMD?

    Inputs:
      - model_name: new model name to be considered and added if new.
      - running_database: Database (output of launch_db) containing
        info on log likelihood etc.
      - model_lists: output of launch_db. A list of lists containing
        every previously considered model, categorised by dimension.

    Outputs:
      TODO: return True if added; False if previously considered?

    Effect:
      - If model hasn't been considered before,
          Adds a row to running_database containing all columns of those.
    """

    model_id = int(model_id)
    model_name = qmla.database_framework.alph(model_name)
    model_num_qubits = qmla.database_framework.get_num_qubits(model_name)

    if (
        qmla.database_framework.consider_new_model(
            model_lists, model_name, running_database) == 'New'
        or
        force_create_model == True
    ):
        model_lists[model_num_qubits].append(model_name)
        log_print(
            [
                "Model ", model_name,
                "not previously considered -- adding.",
                "ID:", model_id
            ],
            log_file
        )
        op = qmla.database_framework.Operator(
            name=model_name, undimensionalised_name=model_name
        )
        num_rows = len(running_database)

        reduced_qml_instance = qmla.model_instances.ModelInstanceForStorage(
            model_name=model_name,
            model_terms_matrices=op.constituents_operators,
            true_oplist=true_model_terms_matrices,
            true_model_terms_params=true_model_terms_params,
            model_id=int(model_id),
            qid=qid,
            host_name=host_name,
            port_number=port_number,
            log_file=log_file
        )
        running_db_new_row = pd.Series({
            '<Name>': model_name,
            'Status': 'Active',
            'Completed': False,
            'branch_id': int(branch_id),
            'Reduced_Model_Class_Instance': reduced_qml_instance,
            'Operator_Instance': op,
            'Epoch_Start': 0, 
            'ModelID': int(float(model_id)),
        })
        running_database.loc[num_rows] = running_db_new_row
        return True
    else:
        log_print(
            [
                "Model {} previously considered.".format(
                    model_name
                )
            ],
            log_file
        )
        return False
