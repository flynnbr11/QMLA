import numpy as np
import random
import pickle
import argparse
import sys
import os

import matplotlib.pyplot as plt
import pickle

# Find QMLA and import it
p = os.path.abspath(os.path.realpath(__file__))
elements = p.split('/')[:-2]
qmla_root = os.path.abspath('/'.join(elements))
sys.path.append(qmla_root)
import qmla

pickle.HIGHEST_PROTOCOL = 4

# Parse arguments from bash
parser = argparse.ArgumentParser(
    description='Pass files to pickel QHL parameters.'
)

parser.add_argument(
    '-dir', '--run_directory',
    help='Absolute path of directory for this run.',
    type=str,
    default=os.getcwd()
)
parser.add_argument(
    '-prt', '--particle_number',
    help='Number of particles used during training',
    type=float,
    default=100
)
parser.add_argument(
    '-ggr', '--growth_generation_rule',
    help="Growth rule of new models",
    type=str,
    default=0
)
parser.add_argument(
    '-agr', '--alternative_growth_rules',
    help='Growth rules to form other trees.',
    action='append',
    default=[],
)

parser.add_argument(
    '-runinfo', '--run_info_file',
    help="File to pickle true params list to.",
    type=str,
    default=None
)
parser.add_argument(
    '-sysmeas', '--system_measurements_file',
    help='Path to which to save true parameters.',
    type=str,
    default="{}/true_model_terms_params.p".format(os.getcwd())
)

parser.add_argument(
    '-plotprobes', '--probes_plot_file',
    help="File to pickle probes, against which to plot expectation values.",
    type=str,
    default=None
)

parser.add_argument(
    '-log', '--log_file',
    help='File to log QMLA run.',
    type=str,
    default='qmla_run.log'
)

###############
# Set shared QMLA parameters for this run
###############

arguments = parser.parse_args()

num_particles = arguments.particle_number
growth_generation_rule = arguments.growth_generation_rule
log_file = arguments.log_file
probes_plot_file = arguments.probes_plot_file
run_directory = arguments.run_directory

# Generate GR instances
growth_class_attributes = {
    'true_params_path' : arguments.run_info_file,
    'plot_probes_path' : probes_plot_file,
    'log_file': log_file
}

all_growth_rules = [growth_generation_rule]
alternative_growth_rules = arguments.alternative_growth_rules
all_growth_rules.extend(alternative_growth_rules)
all_growth_rules = list(set(all_growth_rules))

unique_growth_classes = {
    gr : qmla.get_growth_generator_class(
            growth_generation_rule=gr,
            **growth_class_attributes
    ) for gr in all_growth_rules
}

growth_class = unique_growth_classes[growth_generation_rule]
probe_max_num_qubits_all_growth_rules = max([
    gr.max_num_probe_qubits for gr in unique_growth_classes.values()
])

# Use setup so far to generate parameters 
true_model = growth_class.true_model

qmla.set_shared_parameters(
    growth_class = growth_class,
    run_info_file = arguments.run_info_file,
    all_growth_rules = all_growth_rules,
    run_directory = run_directory,
    num_particles = num_particles,
    generate_evaluation_experiments = True, 
    probe_max_num_qubits_all_growth_rules = probe_max_num_qubits_all_growth_rules, 
)


# Probes used for plotting by all instances in this run, for consistency.
plot_probe_dict = growth_class.plot_probe_generator(
    probe_maximum_number_qubits = probe_max_num_qubits_all_growth_rules, 
)
pickle.dump(
    plot_probe_dict,
    open(probes_plot_file, 'wb')
)

# Store growth rule config to share with all instances in this run
path_to_store_configs = os.path.join(
    run_directory, 
    'configs_growth_rules.p'
)
growth_rule_configurations = {
    gr : unique_growth_classes[gr].store_growth_rule_configuration()
    for gr in unique_growth_classes
}
pickle.dump(
    growth_rule_configurations,
    open(path_to_store_configs, 'wb')
)

# Get system measurements
# i.e. compute them only once and share with all instances
true_system_measurements = growth_class.get_measurements_by_time()
pickle.dump(
    true_system_measurements,
    open(
        arguments.system_measurements_file, 'wb'
    )
)

# Store an example of the probes used
growth_class.generate_probes(
    probe_maximum_number_qubits = probe_max_num_qubits_all_growth_rules, 
    noise_level=0, # TODO get directly in GR
    minimum_tolerable_noise=0.0,
)

probes_dir = os.path.join(
    run_directory,
    "training_probes"
)
try:
    os.makedirs(probes_dir)
    system_probes_path = os.path.join(
        probes_dir,
        "system_probes.p"
    )
    pickle.dump(
        growth_class.probes_system,
        open(system_probes_path, 'wb')
    )
    simulator_probes_path = os.path.join(
        probes_dir,
        "simulator_probes.p"
    )
    pickle.dump(
        growth_class.probes_simulator,
        open(simulator_probes_path, 'wb')
    )
except:
    # Something already stored as example
    pass
