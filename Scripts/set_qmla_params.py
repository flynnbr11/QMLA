import numpy as np
import random
import pickle
import argparse
import sys
import os
import matplotlib.pyplot as plt

sys.path.append("..")
import qmla
pickle.HIGHEST_PROTOCOL = 4

# Parse arguments from bash
parser = argparse.ArgumentParser(
    description='Pass files to pickel QHL parameters.'
)

parser.add_argument(
    '-true', '--true_params_file',
    help="File to pickle true params list to.",
    type=str,
    default=None
)
parser.add_argument(
    '-rand_t', '--random_true_params',
    help="Bool: use random true parameters or those defined in this file.",
    type=int,
    default=0
)

parser.add_argument(
    '-prior', '--prior_file',
    help="File to pickle prior specific terms to.",
    type=str,
    default=None
)

parser.add_argument(
    '-rand_p', '--random_prior_terms',
    help="Bool: use random true parameters or those defined in this file.",
    type=int,
    default=0
)

parser.add_argument(
    '-exp', '--use_experimental_data',
    help="Bool: use experimental data or not.",
    type=int,
    default=0
)

parser.add_argument(
    '-ggr', '--growth_generation_rule',
    help="Generator of new models",
    type=str,
    default=0
)

parser.add_argument(
    '-agr', '--alternative_growth_rules',
    help='Growth rules to form other trees.',
    # type=str,
    action='append',
    default=[],
)

parser.add_argument(
    '-log', '--log_file',
    help='File to log RQ workers.',
    type=str,
    default='qmd.log'
)

parser.add_argument(
    '-dir', '--results_directory',
    help='Relative directory to store results in.',
    type=str,
    default=os.getcwd()
)

parser.add_argument(
    '-probe', '--probes_plot_file',
    help="File to pickle probes against which to plot expectation values.",
    type=str,
    default=None
)

parser.add_argument(
    '-sp', '--special_probe',
    help="Special type of probe, e.g. |+>, or ideal (sum of eigenstates).",
    type=str,
    default=None
)

parser.add_argument(
    '-plus', '--force_plus_probe',
    help="Whether to enforce plots to use |+>^n as probe.",
    type=int,
    default=0
)

parser.add_argument(
    '-g', '--gaussian',
    help="Whether to use normal (Gaussian) distribution. If False: uniform.",
    type=int,
    default=1
)
parser.add_argument(
    '-true_expec_path', '--true_expec_path',
    help='Path to save true params to.',
    type=str,
    default="{}/true_model_terms_params.p".format(os.getcwd())
)
parser.add_argument(
    '-min', '--param_min',
    help="Minimum valid parameter value",
    type=float,
    default=0.0
)
parser.add_argument(
    '-max', '--param_max',
    help="Maximum valid parameter value",
    type=float,
    default=1.0
)

parser.add_argument(
    '-mean', '--param_mean',
    help="Default mean of normal distribution for unspecified parameters.",
    type=float,
    default=0.5
)
parser.add_argument(
    '-sigma', '--param_sigma',
    help="Default sigma of normal distribution for unspecified parameters.",
    type=float,
    default=0.25
)

parser.add_argument(
    '-pnoise', '--probe_noise_level',
    help='Noise level to add to probe for learning',
    type=float,
    default=0.03
)

parser.add_argument(
    '-prt', '--particle_number',
    help='Number of particles used during training',
    type=float,
    default=100
)

print("Set QMLA Params script")

arguments = parser.parse_args()
random_true_params = bool(arguments.random_true_params)
random_prior = bool(arguments.random_prior_terms)
# exp_data = bool(arguments.use_experimental_data)
num_particles = arguments.particle_number
growth_generation_rule = arguments.growth_generation_rule
log_file = arguments.log_file
probes_plot_file = arguments.probes_plot_file
results_directory = arguments.results_directory

growth_class_attributes = {
    # 'use_experimental_data': exp_data,
    'true_params_path' : arguments.true_params_file,
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
probe_max_num_qubits_all_growth_rules = max( 
    [
        gr.max_num_probe_qubits for gr in unique_growth_classes.values()
    ]
)

true_model = growth_class.true_model
force_plus_probe = bool(arguments.force_plus_probe)
probe_noise_level = arguments.probe_noise_level

true_prior_plot_file = str(
    results_directory +
    '/prior_true_params.png'
)

true_prior = growth_class.get_prior(
    model_name=true_model,
    log_file=log_file,
    log_identifier='[Set QMLA params script]'
)
prior_data = {
    'true_prior': true_prior
}

pickle_file = arguments.prior_file
if pickle_file is not None:
    import pickle
    pickle.dump(
        prior_data,
        open(pickle_file, 'wb')
    )


if arguments.true_params_file is not None:
    qmla.set_shared_parameters(
        growth_class=growth_class,
        true_prior=true_prior,
        pickle_file=arguments.true_params_file,
        all_growth_rules=all_growth_rules,
        # exp_data=exp_data,
        results_directory=results_directory,
        num_particles = num_particles,
        generate_evaluation_experiments=True, 
        probe_max_num_qubits_all_growth_rules = probe_max_num_qubits_all_growth_rules, 
        true_prior_plot_file=true_prior_plot_file
    )


###
# Now generate a probe dict to be used by all instances
# of QMD within this run, when plotting results.
# Store it in the provided argument, plot_probe_dict.
###

print("Generating probe dict for plotting")

# TODO
plot_probe_dict = growth_class.plot_probe_generator(
    true_model=true_model,
    growth_generator=growth_generation_rule,
    probe_maximum_number_qubits = probe_max_num_qubits_all_growth_rules, 
    # experimental_data=exp_data,
    noise_level=probe_noise_level,
)

# for k in list(plot_probe_dict.keys()):
#     # replace tuple like key returned, with just dimension.
#     plot_probe_dict[k[1]] = plot_probe_dict.pop(k)
if probes_plot_file is not None:
    import pickle
    pickle.dump(
        plot_probe_dict,
        open(probes_plot_file, 'wb')
    )

# store growth rule config to share with all instances in this run
path_to_store_configs = os.path.join(
    results_directory, 
    'growth_rule_configs.p'
)
growth_rule_configurations = {
    gr : unique_growth_classes[gr].store_growth_rule_configuration()
    for gr in unique_growth_classes
}
pickle.dump(
    growth_rule_configurations,
    open(path_to_store_configs, 'wb')
)
# get measurements of the true system
## work them out only once and share with all instances
print("[Set QMLA params] Storing true measurements to {}".format(
    arguments.true_expec_path
    )
)

true_system_measurements = growth_class.get_measurements_by_time()
pickle.dump(
    true_system_measurements,
    open(
        arguments.true_expec_path,
        'wb'
    )
)

# store an example of the probes used
growth_class.generate_probes(
    probe_maximum_number_qubits = probe_max_num_qubits_all_growth_rules, 
    # experimental_data=exp_data,
    noise_level=growth_class.probe_noise_level,
    minimum_tolerable_noise=0.0,
)

probes_dir = str(
    results_directory
    + 'training_probes/'
)
try:
    os.makedirs(probes_dir)
    print("QMLA SETTINGS - storing probes sample to ", probes_dir)
    system_probes_path = os.path.join(
        probes_dir
        + 'system_probes.p'
    )
    pickle.dump(
        growth_class.probes_system,
        open(system_probes_path, 'wb')
    )
    simulator_probes_path = os.path.join(
        probes_dir
        + 'simulator_probes.p'
    )
    pickle.dump(
        growth_class.probes_simulator,
        open(simulator_probes_path, 'wb')
    )
except:
    # something already stored as example
    pass
