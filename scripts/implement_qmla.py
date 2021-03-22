import os as os
import warnings
import numpy as np
import itertools as itr
import matplotlib.pyplot as plt
import sys as sys
import pandas as pd
import warnings
import time as time
import logging
import random
import pickle
pickle.HIGHEST_PROTOCOL = 4

p = os.path.abspath(os.path.realpath(__file__))
elements = p.split('/')[:-2]
qmla_root = os.path.abspath('/'.join(elements))
sys.path.append(qmla_root)

import qmla
from qmla import model_building_utilities
from qmla.quantum_model_learning_agent import QuantumModelLearningAgent  # QMD class in Library
from qmla import redis_settings as rds
import qmla.logging

#########################
# Parse input variables to use in QMD; store in class qmla_controls.
#########################
qmla_controls = qmla.parse_cmd_line_args(sys.argv[1:])
exploration_class = qmla_controls.exploration_class


#########################
# Set up
#########################
def log_print(to_print_list, log_file=None):
    qmla.logging.print_to_log(
        to_print_list=to_print_list,
        log_file=qmla_controls.log_file,
        log_identifier='Implement QMLA script'
    )    

# TODO replace custom logger with default python one as below:
# logging.basicConfig(
#     filename=qmla_controls.alt_log_file, 
#     level=logging.INFO,
#     format='%(levelname)s:%(asctime)s:%(message)s'
# )
# logging.info('Implement QMLA script')


print("Implement QMLA script")
start = time.time()

experimental_measurements_dict = pickle.load(
    open(qmla_controls.system_measurements_file, 'rb')
)
model_priors = None
results_directory = qmla_controls.results_directory
long_id = qmla_controls.long_id

if qmla_controls.further_qhl == True:
    # TODO further QHL stage out of date and won't work with 
    # new code -- update
    # make an exploration strategy accepting list of models and priors?

    models_scores = str(
        qmla_controls.results_directory
        + 'average_priors.p'
    )
    print("QMLA results CSV in ", models_scores)
    model_priors = pickle.load(
        open(
            models_scores,
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

qmla_instance = QuantumModelLearningAgent(
    qmla_controls=qmla_controls,
    model_priors=model_priors,
    experimental_measurements=experimental_measurements_dict,
)

if qmla_controls.qhl_mode:
    qmla_instance.run_quantum_hamiltonian_learning()

elif (
    qmla_controls.further_qhl
    or qmla_controls.qhl_mode_multiple_models
):

    if qmla_controls.qhl_mode_multiple_models:
        qhl_models = exploration_class.qhl_models
        output_prefix = ''  # TODO make so that this can have an output prefix
    else:
        qhl_models = further_qhl_models
        output_prefix = 'further_qhl_'

    log_print([
        "Launching QHL with multiple models: {}".format(qhl_models)
    ])
    qmla_instance.run_quantum_hamiltonian_learning_multiple_models(
        model_names = qhl_models
    )

else:
    qmla_instance.run_complete_qmla()

#########################
# Analyse instance after running
# TODO: wrap these functions into qmla_instance analysis method
#########################

if qmla_controls.pickle_qmla_instance:
    log_print([
        "QMLA complete. Pickling result to {}".format(
            qmla_controls.class_pickle_file
        )
    ])
    qmla_instance._delete_unpicklable_attributes() # TODO call from within QMLA
    with open(qmla_controls.class_pickle_file, "wb") as pkl_file:
        pickle.dump(qmla_instance, pkl_file, protocol=4)

qmla_instance.analyse_instance()
qmla_instance.plot_instance_outcomes()

#########################
# Wrap up
#########################
end = time.time()
log_print(["Time taken:", end - start])
log_print(["QMLA timings:", qmla_instance.timings, "\nCalls:", qmla_instance.call_counter])
log_print([
    "END of QMLA id", qmla_controls.qmla_id, ":",
    qmla_controls.num_particles, " particles;",
    qmla_controls.num_experiments, "experiments.",
    ". Time:", end - start
])

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
