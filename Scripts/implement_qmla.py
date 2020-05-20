import os as os
import warnings
import numpy as np
import itertools as itr
import matplotlib.pyplot as plt
import sys as sys
import pandas as pd
import warnings
import time as time
import random
import pickle
pickle.HIGHEST_PROTOCOL = 4


p = os.path.abspath(os.path.realpath(__file__))
elements = p.split('/')[:-2]
qmla_root = os.path.abspath('/'.join(elements))
sys.path.append(qmla_root)

# sys.path.append("..")
import qmla
from qmla import database_framework
from qmla.quantum_model_learning_agent import QuantumModelLearningAgent  # QMD class in Library
from qmla import redis_settings as rds
import qmla.logging

#########################
# Parse input variables to use in QMD; store in class qmla_controls.
#########################
qmla_controls = qmla.parse_cmd_line_args(sys.argv[1:])
growth_class = qmla_controls.growth_class


#########################
# Set up
#########################
def log_print(to_print_list, log_file=None):
    qmla.logging.print_to_log(
        to_print_list=to_print_list,
        log_file=qmla_controls.log_file,
        log_identifier='Implement QMLA script'
    )    

print("Implement QMLA script")
start = time.time()

experimental_measurements_dict = pickle.load(
    open(qmla_controls.true_expec_path, 'rb')
)
model_priors = None
results_directory = qmla_controls.results_directory
long_id = qmla_controls.long_id

if qmla_controls.further_qhl == True:
    # TODO further QHL stage out of date and won't work with 
    # new code -- update
    # make a growth rule accepting list of models and priors?

    qmd_results_model_scores_csv = str(
        qmla_controls.results_directory
        + 'average_priors.p'
    )
    print("QMLA results CSV in ", qmd_results_model_scores_csv)
    model_priors = pickle.load(
        open(
            qmd_results_model_scores_csv,
            'rb'
        )
    )
    log_print(
        ["Futher QHL. Model_priors:\n", model_priors],
    )
    first_layer_models = list(model_priors.keys())
    further_qhl_models = list(model_priors.keys())


#########################
# Run QMLA mode specified in launch scipt
#########################
print("------ QMLA starting ------")

# working on dev version
# qmla_instance = qmla.DevQuantumModelLearningAgent(
qmla_instance = QuantumModelLearningAgent(
    qmla_controls=qmla_controls,
    model_priors=model_priors,
    experimental_measurements=experimental_measurements_dict,
)


if qmla_controls.qhl_mode:
    qmla_instance.run_quantum_hamiltonian_learning()

elif (
    qmla_controls.further_qhl == True
    or qmla_controls.qhl_mode_multiple_models == True
):

    if qmla_controls.qhl_mode_multiple_models == True:
        qhl_models = growth_class.qhl_models
        output_prefix = ''  # TODO make so that this can have an output prefix
    else:
        qhl_models = further_qhl_models
        output_prefix = 'further_qhl_'

    log_print(
        [
            "Launching QHL with multiple models: {}".format(qhl_models)
        ],
    )
    qmla_instance.run_quantum_hamiltonian_learning_multiple_models(
        model_names=qhl_models
    )
else:
    qmla_instance.run_complete_qmla()


#########################
# Analyse instance after running
# TODO: wrap these functions into qmla_instance analysis method
#########################
if qmla_controls.qhl_mode:
    log_print(
        [
            "QHL complete",
        ],
    )
    if qmla_controls.pickle_qmd_class:
        log_print(
            [
                "QMD complete. Pickling result to",
                qmla_controls.class_pickle_file
            ], 
        )
        qmla_instance._delete_unpicklable_attributes() # TODO call from within QMLA
        with open(qmla_controls.class_pickle_file, "wb") as pkl_file:
            pickle.dump(qmla_instance, pkl_file, protocol=4)

    if qmla_controls.save_plots:
        try:
            log_print(
                [
                    "Plotting parameter estimates",
                ]
            )
            qmla_instance.plot_parameter_learning_single_model(
                true_model=True,
                save_to_file=str(
                    qmla_controls.plots_directory +
                    'qhl_parameter_estimates_' +
                    str(qmla_controls.long_id) +
                    '.png'
                )
            )
        except BaseException:
            pass

        try:
            log_print(
                [
                    "Plotting volumes",
                ],
            )
            qmla_instance.plot_volume_after_qhl(
                save_to_file=str(
                    qmla_controls.plots_directory +
                    'qhl_volume_' +
                    str(qmla_controls.long_id) +
                    '.png'
                )
            )
        except BaseException:
            pass
        log_print(
            [
                "Plotting Quadratic Losses",
            ],
        )

        qmla_instance.plot_branch_champs_quadratic_losses(
            save_to_file=str(
                qmla_controls.plots_directory +
                'qhl_quadratic_loss_'
                + str(qmla_controls.long_id) + '.png'
            )
        )

        true_mod_instance = qmla_instance.get_model_storage_instance_by_id(
            qmla_instance.true_model_id
        )

    # log_print(
    #     [
    #         "Plotting Dynamics",
    #     ],
    # )
    qmla_instance.plot_branch_champions_dynamics(
        save_to_file=str(
            qmla_controls.plots_directory +
            'dynamics_' +
            str(qmla_controls.long_id) +
            '.png'
        )
    )
    # log_print(
    #     [
    #         "Finished plotting dynamics",
    #     ],
    # )

    # true_mod = qmla_instance.get_model_storage_instance_by_id(
    #     qmla_instance.true_model_id
    # )

    results_file = qmla_controls.results_file
    pickle.dump(
        # qmla_instance.champion_results,
        qmla_instance.get_results_dict(),
        open(results_file, "wb"),
        protocol=4
    )

elif (
    qmla_controls.further_qhl == True
    or qmla_controls.qhl_mode_multiple_models == True
):
    model_ids = [
        database_framework.model_id_from_name(
            db=qmla_instance.model_database,
            name=mod
            # ) for mod in further_qhl_models
        ) for mod in qhl_models
    ]

    qmla_instance.plot_branch_champions_dynamics(
        save_to_file=str(
            qmla_controls.plots_directory +
            'dynamics_' +
            str(qmla_controls.long_id) +
            '.png'
        )
    )

    if qmla_controls.pickle_qmd_class:
        log_print(
            [
                "QMD complete. Pickling result to",
                qmla_controls.class_pickle_file
            ],
        )
        qmla_instance._delete_unpicklable_attributes() # TODO call from within QMLA
        with open(qmla_controls.class_pickle_file, "wb") as pkl_file:
            pickle.dump(qmla_instance, pkl_file, protocol=4)
    for mid in model_ids:
        mod = qmla_instance.get_model_storage_instance_by_id(mid)
        name = mod.model_name

        results_file = str(
            qmla_controls.results_directory +
            output_prefix +
            'results_' +
            str("m{}_q{}.p".format(
                mid, 
                qmla_controls.long_id
                )
            )
            # + '_' +
            # str(qmla_controls.long_id) +
            # '.p'
        )
        print("[Exp] results file:", results_file)

        pickle.dump(
            # mod.results_dict,
            qmla_instance.get_results_dict(model_id = mid),
            open(results_file, "wb"),
            protocol=4
        )
else:
    print(" \n\n------ QMLA learning stage complete ------\n\n")
    print(" ------ Analysis ------")

    """
    Tidy up and analysis.
    """
    expec_value_mods_to_plot = []
    try:
        expec_value_mods_to_plot = [qmla_instance.true_model_id]
    except BaseException:
        pass
    expec_value_mods_to_plot.append(qmla_instance.champion_model_id)

    print("Plotting expectation values.")
    plot_dynamics_all_models = False
    qmla_instance.plot_branch_champions_dynamics(
        all_models=plot_dynamics_all_models,
        save_to_file=str(
            qmla_controls.plots_directory +
            'dynamics_' +
            str(qmla_controls.long_id) +
            '.png'
        )
    )

    print("Plotting statistical metrics")
    qmla_instance.get_statistical_metrics(
        save_to_file=os.path.join(
            qmla_controls.plots_directory, 
            "metrics_{}.png".format(qmla_controls.long_id)
        )
    )

    champ_mod = qmla_instance.get_model_storage_instance_by_id(
        qmla_instance.champion_model_id
    )

    if qmla_controls.save_plots:
        try:
            print("plot_branch_champs_volumes")
            qmla_instance.plot_branch_champs_volumes(
                save_to_file=str(
                    qmla_controls.plots_directory +
                    'volumes_all_models_' + str(qmla_controls.long_id) + '.png')
            )
            print("plotExpecValues2")
            qmla_instance.plot_branch_champs_volumes(
                branch_champions=True,
                save_to_file=str(qmla_controls.plots_directory +
                                 'volumes_branch_champs_' + str(qmla_controls.long_id) +
                                 '.png')
            )
            print("plotQuadLoss")
            qmla_instance.plot_branch_champs_quadratic_losses(
                save_to_file=str(
                    qmla_controls.plots_directory +
                    'quadratic_loss_' + str(qmla_controls.long_id) +
                    '.png'
                )
            )
        except BaseException:
            print("Couldn't plot all individual QMD plots.")

        true_op_known = False
        try:
            if qmla_instance.true_model_id:
                true_op_known = True
        except BaseException:
            true_op_known = False

        if true_op_known == True:
            # if QMD has knowledge of the "true" model, then plot params
            try:
                print("plot_parameter_learning_single_model")
                qmla_instance.plot_parameter_learning_single_model(
                    model_id=qmla_instance.true_model_id,
                    save_to_file=str(
                        qmla_controls.plots_directory +
                        'true_model_parameter_estimates_' +
                        str(qmla_controls.long_id) +
                        '.png'
                    )
                )
            except BaseException:
                print(
                    "Failed to plot parameter estimates for true model:",
                    qmla_instance.true_model_name
                )

            if qmla_instance.champion_model_id != qmla_instance.true_model_id:
                try:
                    print("plot_parameter_learning_single_model champ id != true id")
                    qmla_instance.plot_parameter_learning_single_model(
                        model_id=qmla_instance.champion_model_id,
                        save_to_file=str(qmla_controls.plots_directory +
                                         'champ_model_parameter_estimates_' + str(qmla_controls.long_id) +
                                         '.png')
                    )
                except BaseException:
                    print(
                        "Failed to plot parameter estimates for ",
                        "champ model:",
                        qmla_instance.champion_model_id
                    )

        else:
            try:
                print("else plot_parameter_learning_single_model")
                qmla_instance.plot_parameter_learning_single_model(
                    model_id=qmla_instance.champion_model_id,
                    save_to_file=str(qmla_controls.plots_directory +
                                     'champ_model_parameter_estimates_' + str(qmla_controls.long_id) +
                                     '.png')
                )
            except BaseException:
                print(
                    "Failed to plot parameter estimates for champ model:",
                    qmla_instance.champion_model_id
                )
        print("store_bayes_factors_to_csv")
        qmla_instance.store_bayes_factors_to_csv(
            save_to_file=str(
                qmla_controls.results_directory +
                'bayes_factors_' + str(qmla_controls.long_id) + '.csv'
            ),
            names_ids='latex'
        )

        # TODO generalise so tree diagram can be used in all cases
        # currently only useful for Ising growth 2 qubits.
        try:
            print("plot_TreeQMLA")
            qmla_instance.plot_TreeQMLA(
                only_adjacent_branches=False,
                save_to_file=str
                (qmla_controls.plots_directory +
                 'tree_diagram_' +
                 str(qmla_controls.long_id) +
                 '.png')
            )
        except BaseException:
            print("Failed to plot tree for ", qmla_controls.long_id)
            raise

        try:
            print("plot_r_squared_by_epoch_for_model_list")
            qmla_instance.plot_r_squared_by_epoch_for_model_list(
                save_to_file=str(
                    qmla_controls.plots_directory +
                    'r_squared_by_epoch_' + str(qmla_controls.long_id) +
                    '.png'
                )
            )
        except BaseException:
            log_print(
                [
                    "Failed to plot R squared vs epoch.",
                    "Probably a problem caused by introducing rescaling",
                    "resources based on num qubits etc"
                ],
            )

    if (
        qmla_controls.pickle_qmd_class
    ):
        log_print(["QMD complete. Pickling result to",
                   qmla_controls.class_pickle_file], 
                  )
        # pickle in cases where true model found
        qmla_instance._delete_unpicklable_attributes() # TODO call from within QMLA
        with open(qmla_controls.class_pickle_file, "wb") as pkl_file:
            pickle.dump(qmla_instance, pkl_file, protocol=4)

    qmla_instance.growth_class.growth_rule_specific_plots(
        save_directory = qmla_controls.plots_directory,
        qmla_id = qmla_controls.long_id
    )

    qmla_instance.store_bayes_factors_to_shared_csv(
        bayes_csv=str(qmla_controls.cumulative_csv)
    )

    results_file = qmla_controls.results_file
    pickle.dump(
        # qmla_instance.champion_results,
        qmla_instance.get_results_dict(),
        open(results_file, "wb"),
        protocol=4
    )
    pickle.dump(
        qmla_instance.storage, 
        open(
            os.path.join(
                qmla_controls.results_directory, 
                'storage_{}.p'.format(qmla_controls.long_id), 
            ), 
            'wb'
        ),
        protocol = 4, 
    )

#########################
# Wrap up
#########################
end = time.time()
log_print(["Time taken:", end - start])
log_print(["QMLA timings:", qmla_instance.timings, "\nCalls:", qmla_instance.call_counter])
log_print(
    [
        "END of QMLA id", qmla_controls.qmla_id, ":",
        qmla_controls.num_particles, " particles;",
        qmla_controls.num_experiments, "experiments.",
        ". Time:", end - start
    ],
)

if qmla_controls.host_name.startswith('node'):
    try:    
        import redis
        redis_server = redis.Redis(
            host=qmla_controls.host_name, 
            port=qmla_controls.port_number
        )
        log_print([
            "Shutting down redis server -- {}:{}".format(
                qmla_controls.host_name, 
                qmla_controls.port_number
            )
        ])
        redis_server.shutdown()
    except:
        log_print([
            "Failed to shut down server {}:{}".format(
                qmla_controls.host_name, 
                qmla_controls.port_number
            )
        ])

print("-----------QMLA finished; results in {} ---------".format(qmla_controls.results_directory))
