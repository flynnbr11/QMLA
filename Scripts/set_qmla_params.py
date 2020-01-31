import numpy as np
import random
import pickle
import argparse
import sys
import os
import matplotlib.pyplot as plt

sys.path.append("..")
import qmla
pickle.HIGHEST_PROTOCOL = 2

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
    '-probe', '--plot_probe_file',
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


arguments = parser.parse_args()
random_true_params = bool(arguments.random_true_params)
random_prior = bool(arguments.random_prior_terms)
exp_data = bool(arguments.use_experimental_data)
growth_generation_rule = arguments.growth_generation_rule
log_file = arguments.log_file
results_directory = arguments.results_directory

growth_class_attributes = {
    'use_experimental_data': exp_data,
    'log_file': log_file
    # 'probe_generator' : probe_set_generation.restore_dec_13_probe_generation,
    # 'test_growth_class_att' : True
}

# growth_class = get_growth_rule.get_growth_generator_class(
growth_class = qmla.get_growth_generator_class(
    growth_generation_rule=growth_generation_rule,
    **growth_class_attributes
    # use_experimental_data = exp_data
)

all_growth_classes = [growth_generation_rule]
alternative_growth_rules = arguments.alternative_growth_rules
all_growth_classes.extend(alternative_growth_rules)
all_growth_classes = list(set(all_growth_classes))

unique_growth_classes = {}
for g in all_growth_classes:
    try:
        # unique_growth_classes[g] = get_growth_rule.get_growth_generator_class(
        unique_growth_classes[g] = qmla.get_growth_generator_class(
            growth_generation_rule=g,
            **growth_class_attributes
        )
    except BaseException:
        unique_growth_classes[g] = None

true_operator = growth_class.true_operator
plot_probe_file = arguments.plot_probe_file
force_plus_probe = bool(arguments.force_plus_probe)
special_probe = arguments.special_probe
gaussian = bool(arguments.gaussian)
param_min = arguments.param_min
param_max = arguments.param_max
param_mean = arguments.param_mean
param_sigma = arguments.param_sigma
probe_noise_level = arguments.probe_noise_level

true_prior_plot_file = str(
    results_directory +
    '/prior_true_params.png'
)

true_prior = growth_class.get_prior(
    model_name=true_operator,
    log_file=log_file,
    log_identifier='[SetQHLParams]'
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
    qmla.create_qhl_params(
        # true_op = arguments.true_op,
        true_op=true_operator,
        true_prior=true_prior,
        pickle_file=arguments.true_params_file,
        growth_generator=growth_generation_rule,
        unique_growth_classes=unique_growth_classes,
        all_growth_classes=all_growth_classes,
        random_vals=random_true_params,
        rand_min=param_min,
        rand_max=param_max,
        exp_data=exp_data,
        growth_class=growth_class,
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
    true_operator=true_operator,
    growth_generator=growth_generation_rule,
    experimental_data=exp_data,
    noise_level=probe_noise_level,
)

for k in list(plot_probe_dict.keys()):
    # replace tuple like key returned, with just dimension.
    plot_probe_dict[k[1]] = plot_probe_dict.pop(k)
if plot_probe_file is not None:
    import pickle
    pickle.dump(
        plot_probe_dict,
        open(plot_probe_file, 'wb')
    )
