import argparse
import os
import sys
sys.path.append("..")
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), '..')
    )
)

import qmla

######


parser = argparse.ArgumentParser(
    description='Pass files to pickel QHL parameters.'
)

parser.add_argument(
    '-dir', '--results_directory',
    help='Absolute path to directory to store results in.',
    type=str,
    default=''
)

parser.add_argument(
    '-out', '--output_file_name',
    help='Absolute path to directory to store results in.',
    type=str,
    default='analysis.pdf'
)


parser.add_argument(
    '-e', '--num_experiments',
    help='Number of experiments to use for the learning process',
    type=int,
    default=0
)
parser.add_argument(
    '-p', '--num_particles',
    help='Number of particles to use for the learning process',
    type=int,
    default=0
)
parser.add_argument(
    '-bt', '--num_times_bayes',
    help='Number of times to consider in Bayes function.',
    type=int,
    default=0
)
parser.add_argument(
    '-nprobes', '--num_probes',
    help='How many probe states in rota for learning parameters.',
    type=int,
    default=20
)

parser.add_argument(
    '-pnoise', '--probe_noise_level',
    help='Noise level to add to probe for learning',
    type=float,
    default=0.03
)
parser.add_argument(
    '-special_probe', '--special_probe_for_learning',
    help='Specify type of probe to use during learning.',
    type=str,
    default=None
)
parser.add_argument(
    '-ggr', '--growth_generation_rule',
    help='Rule applied for generation of new models during QMD. \
    Corresponding functions must be built into model_generation',
    type=str,
    default='Unknown'
)
parser.add_argument(
    '-run_desc', '--run_description',
    help='Short description of this run',
    type=str,
    default='Unknown'
)
parser.add_argument(
    '-git_commit', '--git_commit_hash',
    help='Hash of git commit',
    type=str,
    default=''
)
parser.add_argument(
    '-t', '--num_tests',
    help="Number of complete tests to average over.",
    type=int,
    default=1
)
parser.add_argument(
    '-rt', '--resample_threshold',
    help='Resampling threshold for QInfer.',
    type=float,
    default=0.5
)
parser.add_argument(
    '-ra', '--resample_a',
    help='Resampling a for QInfer.',
    type=float,
    default=0.98
)
parser.add_argument(
    '-pgh', '--pgh_factor',
    help='Resampling threshold for QInfer.',
    type=float,
    default=1.0
)
parser.add_argument(
    '-log', '--log_file',
    help='File to log RQ workers.',
    type=str,
    default='qmd.log'
)

parser.add_argument(
    '-qhl', '--qhl_test',
    help="Bool to test QHL on given true operator only.",
    type=int,
    default=0
)
parser.add_argument(
    '-mqhl', '--multiQHL',
    help='Run QHL test on multiple (provided) models.',
    type=int,
    default=0
)
parser.add_argument(
    '-cb', '--cumulative_csv',
    help='CSV to store Bayes factors of all QMDs.',
    type=str,
    default='Unknown'
)

parser.add_argument(
    '-exp', '--experimental_data',
    help='Use experimental data if provided',
    type=int,
    default=0
)

arguments = parser.parse_args()
results_directory = arguments.results_directory
output_file_name = arguments.output_file_name

growth_generation_rule = arguments.growth_generation_rule
growth_class = qmla.get_growth_generator_class(
    growth_generation_rule=growth_generation_rule,
    use_experimental_data=arguments.experimental_data,
    log_file=arguments.log_file
)

variables = vars(arguments)
# and some others arguments not explicitly set in launch script

# variables['measurement_type'] = growth_class.measurement_type
variables['expectation_value_func'] = growth_class.expectation_value_function.__name__
variables['heuristic'] = growth_class.model_heuristic_function.__name__
variables['probe_generation_function'] = growth_class.probe_generation_function.__name__
variables['plot_probe_generation_function'] = growth_class.plot_probe_generation_function.__name__

qmla.analysis.combine_analysis_plots(
    results_directory=results_directory,
    output_file_name=output_file_name,
    variables=variables
)
