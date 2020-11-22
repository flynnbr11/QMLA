import sys
import os
import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np

sys.path.append("..")
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), '..')
    )
)
import qmla

parser = argparse.ArgumentParser(
    description='Pass variables for (I)QLE.'
)

# Add parser arguments, ie command line arguments for QMD
# QMLA parameters -- fundamentals such as number of particles etc
parser.add_argument(
    '-es', '--exploration_rules',
    help='Rule applied for generation of new models during QMD. \
    Corresponding functions must be built into model_generation',
    type=str,
    default='two_qubit_ising_rotation_hyperfine'
)
parser.add_argument(
    '-use_aes', '--use_alternative_exploration_strategies',
    help='Whether to use the alternative exploration strategies provided.',
    type=int,
    default=1
)
parser.add_argument(
    '-aes', '--alternative_exploration_strategies',
    help='Exploration Strategies to form other trees.',
    # type=str,
    action='append',
    default=[],
)

parser.add_argument(
    '-e', '--num_experiments',
    help='Number of experiments to use for the learning process',
    type=int,
    default=100
)
parser.add_argument(
    '-p', '--num_particles',
    help='Number of particles to use for the learning process',
    type=int,
    default=100
)
parser.add_argument(
    '-bt', '--num_bayes_times',
    help='Number of times to consider in Bayes function.',
    type=int,
    default=100
)

parser.add_argument(
    '-proc', '--num_processes',
    help='Number of processes available for parallel.',
    type=int,
    default=100
)

parser.add_argument(
    '-res', '--resource_reallocation',
    help='Bool: whether resources are reallocated based on number qubits/terms.',
    type=int,
    default=0
)
parser.add_argument(
    '-scr', '--variable_setting_script',
    help='Script which will be used to source bash env variables from',
    type=str,
    default=None
)

parser.add_argument(
    '-qmdtenv', '--qmd_time_env_var',
    help='Variable to store QMD time value to',
    type=str,
    default="QMD_TIME"
)

parser.add_argument(
    '-qhltenv', '--qhl_time_env_var',
    help='Variable to store QHL time value to',
    type=str,
    default="QHL_TIME"
)

parser.add_argument(
    '-fqhltenv', '--fqhl_time_env_var',
    help='Variable to store FURTHER QHL stage time.',
    type=str,
    default="QHL_TIME"
)

parser.add_argument(
    '-num_proc_env', '--num_processes_env_var',
    help='How many processes to request when running in parallel.',
    type=str,
    default="NUM_PROC"
)


parser.add_argument(
    '-mintime', '--minimum_allowed_time',
    help='Minimum time it is sensible to request',
    type=int,
    default=600
)

parser.add_argument(
    '-time_insurance', '--time_insurance_factor',
    help='Factor to multiple time calculated by, to safely have enough time to finish.',
    type=float,
    default=2
)


# Fill a dictionary of maximum number of models by qubit number/shape
# shape more generally of form (num_qubits, num_terms)

max_num_models_by_shape = {}

# Hamiltonian exponentiation times from testing qutips
# expm function 100 times for each qubit size.
hamiltonian_exponentiation_times = {
    1: 0.0010705208778381348,
    2: 0.0005974793434143067,
    3: 0.0017327165603637695,
    4: 0.0013524317741394043,
    5: 0.004202978610992432,
    6: 0.0029761767387390136,
    7: 0.024223911762237548,
    8: 0.02105050325393677,
    9: 0.08048738956451416,
    10: 0.4869074869155884,
    11: 2.693910768032074
}


# Functions

def time_required(
    exploration_rule,  # ie true growth generator for QHL
    exploration_strategies,
    num_particles,
    num_experiments,
    num_processes=1,
    resource_reallocation=False,
    # num_bayes_times=None,
    minimum_allowed_time=100,
    insurance_factor=2.5,
    **kwargs
):
    times_reqd = {}

    num_hamiltonians_per_model = (
        num_particles *
        (2*num_experiments)
    )

    parallelisability = {}

    print("exploration strategies:", exploration_strategies)
#  print("num models by shape:", generator_max_num_models_by_shape)
    total_time_required = 0
    for gen in exploration_strategies:
        try:
            exploration_class = qmla.get_exploration_class(
                exploration_rules=gen
            )
            generator_max_num_models_by_shape = exploration_class.max_num_models_by_shape
            print("Got time required from growth generator {}".format(gen))
            print(
                "Num models by num qubits:",
                generator_max_num_models_by_shape)
        except BaseException:
            generator_max_num_models_by_shape = max_num_models_by_shape[gen]

        parallelisability[gen] = exploration_class.num_processes_to_parallelise_over
        max_num_qubits = exploration_class.max_num_qubits

        for q in range(1, max_num_qubits + 1):
            time_per_hamiltonian = hamiltonian_exponentiation_times[q]
            try:
                num_models_this_dimension = generator_max_num_models_by_shape[q]
            except BaseException:
                print("num models {} not in generator_max_num_models_by_shape; taking default".format(
                    q
                ))
                num_models_this_dimension = generator_max_num_models_by_shape['other']
            print("GR:", gen, "max num models for ", q, "qubits:",
                  num_models_this_dimension
                  )
            time_this_dimension = (
                num_hamiltonians_per_model *
                time_per_hamiltonian *
                num_models_this_dimension
            )

            total_time_required += time_this_dimension

    total_time_required = (
        insurance_factor * np.round(total_time_required)
    )
    times_reqd['qmd'] = max(
        minimum_allowed_time,
        int(total_time_required)
    )

    # Get time for QHL
    try:
        true_model = exploration_class.true_model
    except BaseException:
        raise
    highest_parallelisability = max(parallelisability.values())
    times_reqd['num_processes'] = highest_parallelisability

    true_dimension = qmla.get_num_qubits(true_model)
    qhl_insurance_factor = 1.5 # buffer to ensure finished in time
    qhl_time = (
        qhl_insurance_factor *
        hamiltonian_exponentiation_times[true_dimension]
        * num_particles
        * num_experiments
    )
    times_reqd['qhl'] = max(
        minimum_allowed_time,
        int(qhl_time)
    )

    # For further qhl, want to account for possibility
    # that winning model is of maximum allowed dimension,
    # so need to request enough time for that case.
    further_qhl_time = (
        hamiltonian_exponentiation_times[max_num_qubits]
        * num_hamiltonians_per_model
    )
    times_reqd['fqhl'] = max(
        minimum_allowed_time,
        int(further_qhl_time)
    )
    for k in ['qmd', 'qhl', 'fqhl']:
        times_reqd[k] *= exploration_class.timing_insurance_factor
        times_reqd[k] =  max(
            minimum_allowed_time,
            int(times_reqd[k])
        )
        times_reqd[k] = int(times_reqd[k])
    print("Time to request:\n", times_reqd)
    return times_reqd


arguments = parser.parse_args()
exploration_rule = arguments.exploration_rules
alternative_exploration_strategies = arguments.alternative_exploration_strategies
all_exploration_strategies = [exploration_rule]
use_alternative_exploration_strategies = bool(
    arguments.use_alternative_exploration_strategies
)
if use_alternative_exploration_strategies == True:
    all_exploration_strategies.extend(alternative_exploration_strategies)
num_particles = arguments.num_particles
num_experiments = arguments.num_experiments
num_bayes_times = arguments.num_bayes_times
num_processes = arguments.num_processes
resource_reallocation = bool(arguments.resource_reallocation)
variable_setting_script = arguments.variable_setting_script
qmd_time_env_var = arguments.qmd_time_env_var
qhl_time_env_var = arguments.qhl_time_env_var
fqhl_time_env_var = arguments.fqhl_time_env_var
num_processes_env_var = arguments.num_processes_env_var
minimum_allowed_time = arguments.minimum_allowed_time
time_insurance_factor = float(arguments.time_insurance_factor)

# print("all exploration strategies:", all_exploration_strategies)
# print("alternative_exploration_strategies:", alternative_exploration_strategies)
time_reqd = time_required(
    exploration_rule=exploration_rule,  # true generator
    exploration_strategies=all_exploration_strategies,
    num_particles=num_particles,
    insurance_factor=time_insurance_factor,
    num_experiments=num_experiments,
    num_processes=num_processes,
    resource_reallocation=resource_reallocation,
    # num_bayes_times=num_bayes_times,
    minimum_allowed_time=minimum_allowed_time,
)


# print(
# 	"Timing heuristic function:",
#   "\nInsurance factor:", time_insurance_factor,
#   "\nuse_alt_exploration_strategies:", use_alternative_exploration_strategies,
# 	"\nQMD:", time_reqd['qmd'],
# 	"\nQHL:", time_reqd['qhl'],
# 	"\nFurtherQHL:", time_reqd['fqhl'],
# )


with open(variable_setting_script, 'a+') as script:
    print(
        "#!/bin/bash\n",
        qmd_time_env_var, "=", time_reqd['qmd'],
        "\n",
        qhl_time_env_var, "=", time_reqd['qhl'],
        "\n",
        fqhl_time_env_var, "=", time_reqd['fqhl'],
        "\n",
        num_processes_env_var, "=", time_reqd['num_processes'],
        sep='',
        file=script
    )
