from __future__ import print_function  # so print doesn't show brackets

import copy
import numpy as np
import time as time
import matplotlib.pyplot as plt

import pickle
pickle.HIGHEST_PROTOCOL = 4
import redis


import qmla.database_framework as database_framework
import qmla.model_instances as QML
import qmla.redis_settings as rds
import qmla.logging

plt.switch_backend('agg')

# Local files

# Single function call, given qmla_core_data and a name, to learn model entirely.

def remote_learn_model_parameters(
    name,
    model_id,
    branch_id,
    growth_generator,
    qmla_core_info_dict=None,
    remote=False,
    host_name='localhost',
    port_number=6379,
    qid=0,
    log_file='rq_output.log'
):
    """
    This is a standalone function to perform QHL on individual
    models  without knowledge of full QMD program.
    QMD info is unpickled from a redis databse, containing
    true operator, params etc.
    Given model names are used to generate ModelInstanceForLearning instances,
    upon which we update the posterior parameter distribution iteratively.
    Once parameters are learned, we pickle the results to dictionaries
    held on a redis database which can be accessed by other actors.

    """

    def log_print(to_print_list):
        qmla.logging.print_to_log(
            to_print_list = to_print_list, 
            log_file = log_file, 
            log_identifier = 'RemoteLearnModel {}'.format(model_id)
        )

    log_print(['Starting for model:', name])
    print("Learning model {}: {}".format( model_id,  name))

    time_start = time.time()

    # Get params from qmla_core_info_dict
    redis_databases = rds.databases_from_qmd_id(host_name, port_number, qid)
    qmla_core_info_database = redis_databases['qmla_core_info_database']
    learned_models_info = redis_databases['learned_models_info']
    learned_models_ids = redis_databases['learned_models_ids']
    active_branches_learning_models = redis_databases['active_branches_learning_models']
    any_job_failed_db = redis_databases['any_job_failed']

    if qmla_core_info_dict is None:
        qmla_core_info_dict = pickle.loads(qmla_core_info_database['qmla_settings'])
        probe_dict = pickle.loads(qmla_core_info_database['ProbeDict'])
    else:  # if in serial, qmla_core_info_dict given, with probe_dict included in it.
        probe_dict = qmla_core_info_dict['probe_dict']

    true_model_terms_matrices = qmla_core_info_dict['true_oplist']
    qhl_plots = qmla_core_info_dict['qhl_plots']
    plots_directory = qmla_core_info_dict['plots_directory']
    long_id = qmla_core_info_dict['long_id']

    # Generate model and learn
    op = database_framework.Operator(name=name)
    model_priors = qmla_core_info_dict['model_priors']
    if (
        model_priors is not None
        and
        database_framework.alph(name) in list(model_priors.keys())
    ):
        prior_specific_terms = model_priors[name]
    else:
        prior_specific_terms = qmla_core_info_dict['prior_specific_terms']

    sim_pars = []
    constituent_terms = database_framework.get_constituent_names_from_name(name)
    for term in op.constituents_names:
        try:
            initial_prior_centre = prior_specific_terms[term][0]
            sim_pars.append(initial_prior_centre)
        except BaseException:
            # if prior not defined, start from 0 for all other params
            initial_prior_centre = 0
            sim_pars.append(initial_prior_centre)

    # add model_db_new_row to model_db and running_database
    # Note: do NOT use pd.df.append() as this copies total DB,
    # appends and returns copy.

    qml_instance = QML.ModelInstanceForLearning(
        name=name,
        # num_probes=num_probes,
        probe_dict=probe_dict,
        qid=qid,
        log_file=log_file,
        model_id=model_id,
        growth_generator=growth_generator,
        model_terms_matrices=op.constituents_operators,
        model_terms_parameters=[sim_pars],
        model_terms_names=op.constituents_names,
        # debug_directory=debug_directory,
        host_name=host_name,
        port_number=port_number,
    )

    log_print(["Starting model QHL update."])
    try:
        update_timer_start = time.time()
        qml_instance.update_model()
        log_print(
            [
                "Time for update alone: {}".format(
                    time.time() - update_timer_start
                )
            ]
        )
    except NameError:
        log_print(
            [
                "QHL failed for model id {}. Setting job failure database_framework.".format(
                    model_id)
            ]
        )
        any_job_failed_db.set('Status', 1)
    except BaseException:
        log_print(
            [
                "QHL failed for model id {}. Setting job failure database_framework.".format(
                    model_id)
            ]
        )
        any_job_failed_db.set('Status', 1)

    if qhl_plots:
        log_print(["Drawing plots for QHL"])
        try:
            if len(true_model_terms_matrices) == 1:  # TODO buggy
                qml_instance.plot_distribution_progression(
                    save_to_file=str(
                        plots_directory
                        + 'qhl_distribution_progression_' + str(long_id) + '.png')
                )

                qml_instance.plot_distribution_progression(
                    renormalise=False,
                    save_to_file=str(
                        plots_directory
                        + 'qhl_distribution_progression_uniform_' + str(long_id) + '.png')
                )
        except BaseException:
            pass

    # only need to store results; throw away class 
    updated_model_info = copy.deepcopy(
        qml_instance.learned_info_dict()
    )
    del qml_instance

    compressed_info = pickle.dumps(
        updated_model_info,
        protocol=4
    )
    # TODO is there a way to use higher protocol when using python3 for faster
    # pickling? this seems to need to be decoded using encoding='latin1'....
    # not entirely clear why this encoding is used
    try:
        learned_models_info.set(
            str(model_id),
            compressed_info
        )
        log_print(
            [
                "Redis learned_models_info added to db for model:",
                str(model_id)
            ]
        )
    except BaseException:
        log_print(
            [
                "Failed to add learned_models_info for model:",
                model_id
            ]
        )
    active_branches_learning_models.incr(int(branch_id), 1)
    time_end = time.time()
    log_print(["Redis SET learned_models_ids:", model_id, "; set True"])
    learned_models_ids.set(str(model_id), 1)

    if remote:
        del updated_model_info
        del compressed_info
        log_print(["Learned. rq time:", str(time_end - time_start)])
        return None
    else:
        return updated_model_info
