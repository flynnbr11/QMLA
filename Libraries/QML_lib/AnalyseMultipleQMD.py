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


from multiQMD_plots import * 



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
results_collection_file = "{}/collect_analyses.p".format(
    directory_to_analyse
)



if true_params_path is not None:
    true_params_info = pickle.load(
        open(
            true_params_path, 
            'rb'
        )
    )
    true_params_dict = true_params_info['params_dict']
    true_operator = true_params_info['true_op']
else:
    true_params_dict = None
    true_operator = true_growth_class.true_operator
true_operator_latex = true_growth_class.latex_name(
    true_operator
)


if exp_data is False:
    name = true_params_info['true_op']
    terms = DataBase.get_constituent_names_from_name(name)
    params=[]
    ops=[]
    for t in terms:
        params.append(true_params_dict[t])
        ops.append( DataBase.compute(t) )
        
    true_ham = np.tensordot(params, ops, axes=1)




#######################################
### Now analyse the results. 
#######################################
print("\nAnalysing and storing results in", directory_to_analyse)

if not directory_to_analyse.endswith('/'):
    directory_to_analyse += '/'

print("Counting model occurences.")

try:
    count_model_occurences(
        latex_map = latex_mapping_file,
        true_operator_latex = true_growth_class.latex_name(
            true_operator
        ),
        save_counts_dict = str(
            directory_to_analyse + 
            "count_model_occurences.p"
        ),
        save_to_file = str(
            directory_to_analyse + 
            "occurences_of_models.png"
        )
    )
except:
    print("Failed to plot # occurences of each model.")
    # raise



if further_qhl_mode==True:
    print("further_qhl_model = True.")

    results_csv_name = 'summary_further_qhl_results.csv'
    results_csv = directory_to_analyse+results_csv_name
    results_file_name_start='further_qhl_results'
    ptq.summariseResultsCSV(
        directory_name=directory_to_analyse, 
        results_file_name_start=results_file_name_start,
        csv_name=results_csv
    )
    plot_desc='further_'

else:
    results_csv_name = 'summary_results.csv'
    results_csv = directory_to_analyse+results_csv_name
    results_file_name_start='results'
    ptq.summariseResultsCSV(
        directory_name=directory_to_analyse, 
        results_file_name_start=results_file_name_start,
        csv_name=results_csv
    )
    plot_desc=''

try:
    average_priors = average_parameters(
        results_path=results_csv,
        top_number_models = arguments.top_number_models 
    )

    avg_priors = str(directory_to_analyse+'average_priors.p')

    pickle.dump(
        average_priors,
        open(avg_priors, 'wb'), 
        protocol=2
    )
except:
    raise
    # for compatability with old versions
    pass


os.chdir(directory_to_analyse)
pickled_files = []
for file in os.listdir(directory_to_analyse):
    # if file.endswith(".p") and file.startswith("results"):
    if (
        file.endswith(".p") 
        and 
        file.startswith(results_file_name_start)
    ):
        pickled_files.append(file)

growth_rules = {}
for f in pickled_files:
    fname = directory_to_analyse+'/'+str(f)
    result = pickle.load(open(fname, 'rb'))
    alph = result['NameAlphabetical']
    # expec_values = result['ExpectationValues']

    # if alph in expectation_values_by_name.keys():
    #     expectation_values_by_name[alph].append(expec_values)
    # else:
    #     expectation_values_by_name[alph] = [expec_values]

    if alph not in list(growth_rules.keys()):
        growth_rules[alph] = result['GrowthGenerator']

# unique_growth_rules = list(set(list(growth_rules.values())))
# unique_growth_classes = {}
# for g in unique_growth_rules:
#     try:
#         unique_growth_classes[g] = GrowthRules.get_growth_generator_class(
#             growth_generation_rule = g
#         )
#     except:
#         unique_growth_classes[g] = None
# growth_classes = {}
# for g in list(growth_rules.keys()):
#     try:
#         growth_classes[g] = unique_growth_classes[growth_rules[g]]
#     except:
#         growth_classes[g] = None

unique_growth_classes = {}
unique_growth_rules = true_params_info['all_growth_classes']
for g in unique_growth_rules:
    try:
        unique_growth_classes[g] = GrowthRules.get_growth_generator_class(
            growth_generation_rule = g
        )
    except:
        unique_growth_classes[g] = None


print("[AnalyseMultipleQMD] unique growth classes:", unique_growth_classes)


# first get model scores
model_score_results = get_model_scores(
    directory_name = directory_to_analyse,
    unique_growth_classes = unique_growth_classes,
    # collective_analysis_pickle_file = results_collection_file, 
)
model_scores = model_score_results['scores']
growth_rules = model_score_results['growth_rules']
growth_classes = model_score_results['growth_classes']
unique_growth_classes =model_score_results['unique_growth_classes']
median_coeff_determination = model_score_results['avg_coeff_determination']
f_scores = model_score_results['f_scores']
latex_coeff_det = model_score_results['latex_coeff_det']

# print("Avg coeff if determination", avg_coeff_determination)
print("Average param estimates")


average_parameter_estimates(
    directory_name = directory_to_analyse, 
    results_path = results_csv, 
    top_number_models = arguments.top_number_models,
    results_file_name_start=results_file_name_start,
    growth_generator = growth_generator,
    unique_growth_classes = unique_growth_classes,
    true_params_dict = true_params_dict,
    save_to_file=  str(
        directory_to_analyse + 
        plot_desc + 
        'param_avg.png'
    )
)

# if exp_data:
print("dynamics plots")

analyse_and_plot_dynamics_multiple_models( # average expected values
    directory_name = directory_to_analyse, 
    dataset = dataset, 
    results_path = results_csv,
    use_experimental_data = exp_data, 
    results_file_name_start = results_file_name_start,
    true_expectation_value_path = true_expec_path,
    growth_generator = growth_generator, 
    unique_growth_classes = unique_growth_classes,
    top_number_models = arguments.top_number_models,
    plot_probe_file = plot_probe_file,
    collective_analysis_pickle_file = results_collection_file,
    save_to_file=str(
        directory_to_analyse+
        plot_desc +
        'expec_vals.png'
    )
)
try:
    r_sqaured_average(
        results_path = results_csv,
        growth_class = true_growth_class, 
        top_number_models = arguments.top_number_models,
        growth_classes_by_name = growth_classes,
        save_to_file=  str(
            directory_to_analyse + 
            plot_desc +
            'r_squared_averages.png'
        )
    )
except:
    print(
        "Unable to plot average R^2 over epochs.", 
        "R^2 at each epoch not stored in QMD (method in QML)."
    )

ptq.average_quadratic_losses(
    results_path = results_csv, 
    growth_classes = unique_growth_classes, 
    growth_generator = growth_generator, 
    top_number_models = arguments.top_number_models,
    save_to_file=  str(
        directory_to_analyse + 
        plot_desc +
        'quadratic_losses_avg.png'
    )
)


volume_average(
    results_path = results_csv,
    growth_class = true_growth_class, 
    top_number_models = arguments.top_number_models,
    save_to_file=  str(
        directory_to_analyse + 
        plot_desc +
        'volume_averages.png'
    )
)

all_times_learned_histogram(
    results_path = results_csv,
    top_number_models = arguments.top_number_models,
    save_to_file=  str(
        directory_to_analyse + 
        plot_desc +
        'times_histogram.png'
    )
)



"""

if qhl_mode==True:
    r_squared_plot = str(
        directory_to_analyse + 
        'r_squared_QHL.png'
    )
    ptq.r_squared_plot(
        results_csv_path = results_csv,
        save_to_file = r_squared_plot
    )
"""

if further_qhl_mode == False:
    print("FURTHER QHL=FALSE. PLOTTING STUFF")
    plot_file = directory_to_analyse+'model_wins.png'

    entropy = inf_gain = 0.0
    print("[AnalyseMultipleQMD] f scores before plot scores:", f_scores)
    plot_scores(
        scores = model_scores,
        growth_classes = growth_classes, 
        unique_growth_classes = unique_growth_classes, 
        growth_rules = growth_rules, 
        coefficients_of_determination = median_coeff_determination, 
        coefficient_determination_latex_name = latex_coeff_det,
        f_scores = f_scores, 
        true_operator = true_operator, 
        growth_generator = growth_generator,
        # collective_analysis_pickle_file = results_collection_file, 
        save_file = plot_file
    )


    # results from get_model_scores above
    models = sorted(model_score_results['wins'].keys())
    models.reverse()

    f_score = {
        'title' : 'F-score',
        'res' : [model_score_results['f_scores'][m] for m in models],
        'range' : 'cap_1',
    }
    r_squared = {
        'title' : '$R^2$',
        'res' : [model_score_results['latex_coeff_det'][m] for m in models],
        'range' : 'cap_1',    
    }

    sensitivity = {
        'title' : 'Sensitivity',
        'res' : [model_score_results['sensitivities'][m] for m in models],
        'range' : 'cap_1',
    }
    precision = {
        'title' : 'Precision', 
        'res' : [model_score_results['precisions'][m] for m in models],
        'range' : 'cap_1',
    }
    wins = {
        'title' : '# Wins',
        'res' : [model_score_results['wins'][m] for m in models],
        'range' : 'uncapped',
    }

    to_plot = [wins, f_score, precision, sensitivity]    

    plot_statistics(
        to_plot, 
        models,
        true_operator = true_operator_latex,
        save_to_file = str(
            directory_to_analyse + 
            'model_stats.png'
        )
    )

    try:
        ptq.plotTrueModelBayesFactors_IsingRotationTerms(
            results_csv_path = all_bayes_csv,
            # correct_mod='xTiPPyTiPPzTiPPxTxPPyTyPPzTz',
            correct_mod = true_operator,
            growth_generator = growth_generator,
            save_to_file = str(
                directory_to_analyse+
                'bayes_comparisons_true_model.png'
            )
        )
    except:
        print("Could not plot histogram of Bayes factors for True model.")
        # raise
    param_plot = str(directory_to_analyse+'sweep_param_total.png')
    param_percent_plot = str(directory_to_analyse+'sweep_param_percentage.png')

    try:
        parameter_sweep_analysis(
            directory_name = directory_to_analyse, 
            results_csv=results_csv, 
            save_to_file=param_plot)
        parameter_sweep_analysis(
            directory_name = directory_to_analyse,
            results_csv=results_csv,
            use_log_times=True,
            use_percentage_models=True, 
            save_to_file=param_percent_plot
        )
    except:
        print("Parameter sweep analysis failed.")
        pass

    do_clustering = True
    if do_clustering:
        try:
            ptq.cluster_results_and_plot(
                path_to_results = results_csv, 
                true_expec_path = true_expec_path, 
                plot_probe_path = plot_probe_file, 
                true_params_path = true_params_path,
                growth_generator = growth_generator, 
                measurement_type = measurement_type, 
                save_param_values_to_file = str(plot_desc + 'clusters_by_param.png'),
                save_param_clusters_to_file = str(plot_desc + 'clusters_by_model.png'),
                save_redrawn_expectation_values = str(plot_desc + 'clusters_expec_vals.png')
            )
        except:
            print("Failed to cluster and replot results.")
            pass
        # raise


    valid_growth_rules_for_multiQMD_tree_plot = [
        'two_qubit_ising_rotation',
        'two_qubit_ising_rotation_hyperfine', 
        'two_qubit_ising_rotation_hyperfine_transverse' ,
        'hyperfine_like'
    ]

    try:
        # if growth_generator in valid_growth_rules_for_multiQMD_tree_plot:

        tree_plot_log = str(directory_to_analyse + 'tree_plot_log.txt')
        sys.stdout = open(
            tree_plot_log, 'w'
        )

        # plot_tree_multi_QMD(
        #     results_csv = results_csv, 
        #     latex_mapping_file=latex_mapping_file, 
        #     avg_type='means', 
        #     all_bayes_csv = all_bayes_csv, 
        #     growth_generator=growth_generator,
        #     entropy = entropy,
        #     inf_gain = inf_gain,
        #     save_to_file='multiQMD_tree_mean_bayes_factors.png'
        # )

        plot_tree_multi_QMD(
            results_csv = results_csv, 
            latex_mapping_file=latex_mapping_file, 
            avg_type='medians', 
            all_bayes_csv = all_bayes_csv, 
            growth_generator=growth_generator,
            entropy = entropy,
            inf_gain = inf_gain,
            save_to_file='multiQMD_tree_median_bayes_factors.png'
        )

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


    except ValueError:
        pass

    except NameError:
        print(
            "Can not plot multiQMD tree -- this might be because only \
            one instance of QMD was performed. All other plots generated \
            without error."
        )
        raise

    except ZeroDivisionError:
        print(
            "Can not plot multiQMD tree -- this might be because only \
            one instance of QMD was performed. All other plots generated \
            without error."
        )
        raise
    except:
        # print("[AnalyseMultipleQMD] Could not plot Multi QMD tree.")
        raise






