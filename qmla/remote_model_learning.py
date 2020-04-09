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

__all__ = [
    'remote_learn_model_parameters'
]

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
    Standalone function to perform Quantum Hamiltonian Learning on individual models.

    Used in conjunction with redis databases so this calculation can be 
        performed without any knowledge of the QMLA instance. 
    Given model ids and names are used to instantiate 
        the ModelInstanceForLearning class, which is then used
        for learning the models parameters.
    QMLA info is unpickled from a redis databse, containing
        true operator, params etc.
    Once parameters are learned, we pickle the results to dictionaries
        held on a redis database which can be accessed by other actors.

    :param str name: model name string
    :param int model_id: unique model id
    :param int branch_id: QMLA branch where the model was generated
    :param str growth_generator: string corresponding to a unique growth rule,
        used by get_growth_generator_class to generate a 
        GrowthRule (or subclass) instance.
    :param dict qmla_core_info_dict: crucial data for QMLA, such as number 
        of experiments/particles etc. Default None: core info is stored on the 
        redis database so can be retrieved there on a server; if running locally, 
        can be passed to save pickling. 
    :param bool remote: whether QMLA is running remotely via RQ workers.
    :param str host_name: name of host server on which redis database exists.
    :param int port_number: this QMLA instance's unique port number,
        on which redis database exists. 
    :param int qid: QMLA id, unique to a single instance within a run. 
        Used to identify the redis database corresponding to this instance.
    :param str log_file: Path of the log file.
    """

    def log_print(to_print_list):
        qmla.logging.print_to_log(
            to_print_list = to_print_list, 
            log_file = log_file, 
            log_identifier = 'RemoteLearnModel {}'.format(model_id)
        )

    log_print(['Starting for model:', name])
    print("Learning model {}: {}".format( model_id,  name))
    log_print([
        "Model {} on branch {}".format(
            model_id, 
            branch_id
        )
    ])

    time_start = time.time()

    # Get params from qmla_core_info_dict
    redis_databases = rds.get_redis_databases_by_qmla_id(host_name, port_number, qid)
    qmla_core_info_database = redis_databases['qmla_core_info_database']
    learned_models_info_db = redis_databases['learned_models_info_db']
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

    # Generate model instance and learn
    qml_instance = QML.ModelInstanceForLearning(
        model_id=model_id,
        model_name=name,
        qid=qid,
        log_file=log_file,
        growth_generator=growth_generator,
        host_name=host_name,
        port_number=port_number,
    )
    evaluation_times = list(np.arange(0, 10, 0.05))
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
        log_print(["Starting model likelihood calculation."])
        print("Computing log likelihood for mod {}".format(model_id))
        qml_instance.compute_likelihood_after_parameter_learning(
            times = evaluation_times
        )
        log_print([
            "Model evaluation ll:", qml_instance.evaluation_log_likelihood
        ])
    except NameError:
        log_print(
            [
                "QHL failed for model id {}. Setting job failure database_framework.".format(
                    model_id)
            ]
        )
        any_job_failed_db.set('Status', 1)
        raise
    except BaseException:
        log_print(
            [
                "QHL failed for model id {}. Setting job failure database_framework.".format(
                    model_id)
            ]
        )
        any_job_failed_db.set('Status', 1)
        raise

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
    try:
        learned_models_info_db.set(
            str(model_id),
            compressed_info
        )
        log_print(
            [
                "Redis learned_models_info_db added to db for model:",
                str(model_id)
            ]
        )
    except BaseException:
        log_print(
            [
                "Failed to add learned_models_info_db for model:",
                model_id
            ]
        )
    active_branches_learning_models.incr(int(branch_id), 1)
    time_end = time.time()
    log_print(["Redis SET learned_models_ids:", model_id, "; set True"])
    log_print(
        [
            "Redis SET active_branches_learning_models:", 
            branch_id, "; now: ",
            active_branches_learning_models.get(int(branch_id))
    ])
    learned_models_ids.set(str(model_id), 1)

    if remote:
        del updated_model_info
        del compressed_info
        log_print(["Learned. rq time:", str(time_end - time_start)])
        return None
    else:
        return updated_model_info
