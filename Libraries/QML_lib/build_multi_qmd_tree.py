import sys, os
import pickle
import matplotlib.pyplot as plt
import pandas
plt.switch_backend('agg')
from matplotlib.lines import Line2D

import argparse
import numpy as np

import DataBase
import PlotQMD as ptq
import ModelNames
# import UserFunctions 
import GrowthRules

global test_growth_class_implementation
test_growth_class_implementation = True




parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')

# Add parser arguments, ie command line arguments for QMD
## QMD parameters -- fundamentals such as number of particles etc
parser.add_argument(
    '-dir', '--results_directory', 
    help="Directory where results of multiple QMD are held.",
    type=str,
    default=os.getcwd()
)

parser.add_argument(
    '-bcsv', '--bayes_csv', 
    help="CSV given to QMD to store all Bayes factors computed.",
    type=str,
    default=os.getcwd()
)

parser.add_argument(
    '-top', '--top_number_models', 
    help="N, for top N models by number of QMD wins.",
    type=int,
    default=3
)

parser.add_argument(
    '-qhl', '--qhl_mode', 
    help="Whether QMD is being used in QHL mode.",
    type=int,
    default=0
)

parser.add_argument(
    '-fqhl', '--further_qhl_mode', 
    help="Whether in further QHL stage.",
    type=int,
    default=0
)
# parser.add_argument(
#     '-data', '--dataset', 
#     help="Which dataset QMD was run using..",
#     type=str,
#     default='NVB_dataset'
# )

parser.add_argument(
    '-params', '--true_params',
    help="Path to pickled true params info.",
    type=str,
    default=None
)


parser.add_argument(
    '-true_expec', '--true_expectation_value_path',
    help="Path to pickled true expectation values.",
    type=str,
    default=None
)

parser.add_argument(
    '-exp', '--use_experimental_data',
    help="Bool: whether or not to use experimental data.",
    type=int,
    default=0
)

parser.add_argument(
    '-ggr', '--growth_generation_rule',
    help='Rule applied for generation of new models during QMD. \
    Corresponding functions must be built into ModelGeneration',
    type=str,
    default=None
)

# parser.add_argument(
#   '-meas', '--measurement_type',
#   help='Which measurement type to use. Must be written in Evo.py.',
#   type=str,
#   default='full_access'
# )

parser.add_argument(
    '-latex', '--latex_mapping_file',
    help='File path to save tuples which give model \
        string names and latex names.',
    type=str,
    default=None
)
parser.add_argument(
  '-plot_probes', '--plot_probe_file', 
  help="File to pickle probes against which to plot expectation values.",
  type=str,
  default=None
)
parser.add_argument(
  '-plus', '--force_plus_probe', 
  help="Whether to enforce plots to use |+>^n as probe.",
  type=int,
  default=0
)




arguments = parser.parse_args()
directory_to_analyse = arguments.results_directory
all_bayes_csv = arguments.bayes_csv
qhl_mode = bool(arguments.qhl_mode)
further_qhl_mode = bool(arguments.further_qhl_mode)
true_params_path = arguments.true_params
exp_data = arguments.use_experimental_data
true_expec_path = arguments.true_expectation_value_path
growth_generator = arguments.growth_generation_rule
true_growth_class = GrowthRules.get_growth_generator_class(growth_generator)
dataset = true_growth_class.experimental_dataset
measurement_type = true_growth_class.measurement_type
latex_mapping_file = arguments.latex_mapping_file
plot_probe_file = arguments.plot_probe_file
force_plus_probe = bool(arguments.force_plus_probe)

results_csv_name = '/summary_results.csv'
results_csv = directory_to_analyse+results_csv_name


def plot_tree_multi_QMD(
        results_csv, 
        all_bayes_csv, 
        latex_mapping_file,
        avg_type='medians',
        growth_generator=None,
        entropy=None, 
        inf_gain=None, 
        save_to_file=None
    ):
    qmd_res = pandas.DataFrame.from_csv(
        results_csv, 
        index_col='LatexName'
    )
    mods = list(qmd_res.index)
    winning_count = {}
    for mod in mods:
        winning_count[mod]=mods.count(mod)

    ptq.cumulativeQMDTreePlot(
        cumulative_csv=all_bayes_csv, 
        wins_per_mod=winning_count,
        latex_mapping_file=latex_mapping_file, 
        growth_generator=growth_generator,
        only_adjacent_branches=False, 
        avg=avg_type, entropy=entropy, inf_gain=inf_gain,
        save_to_file=save_to_file
    )        


# plot_tree_multi_QMD(
#     results_csv = results_csv, 
#     latex_mapping_file=latex_mapping_file, 
#     all_bayes_csv = all_bayes_csv, 
#     growth_generator=growth_generator,
#     avg_type='means',
#     entropy = None,
#     inf_gain = None,
#     save_to_file='multiQMD_tree.png'
# )


tree_plot_log = str(directory_to_analyse + 'tree_plot_log.txt')
sys.stdout = open(
    'tree_plot_log', 'w'
)

plot_tree_multi_QMD(
    results_csv = results_csv, 
    latex_mapping_file=latex_mapping_file, 
    avg_type='medians', 
    all_bayes_csv = all_bayes_csv, 
    growth_generator=growth_generator,
    entropy = None,
    inf_gain = None,
    save_to_file='multiQMD_tree_median_bayes_factors.png'
)
