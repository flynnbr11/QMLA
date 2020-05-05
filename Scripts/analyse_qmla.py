import numpy as np
import argparse
from matplotlib.lines import Line2D
import sys
import os
import pickle
import matplotlib.pyplot as plt
import pandas

plt.switch_backend('agg')
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), '..')
    )
)
import qmla
import qmla.analysis

parser = argparse.ArgumentParser(
    description='Pass variables for QMLA.'
)

# Add parser arguments, ie command line arguments for QMD
# QMLA parameters -- fundamentals such as number of particles etc
parser.add_argument(
    '-dir', '--results_directory',
    help="Directory where results of multiple QMD are held.",
    type=str,
    default=os.getcwd()
)
parser.add_argument(
    '-log', '--log_file',
    help='File to log RQ workers.',
    type=str,
    default='.default_qmd_log.log'
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

parser.add_argument(
    '-params', '--true_model_terms_params',
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
    Corresponding functions must be built into model_generation',
    type=str,
    default=None
)
parser.add_argument(
    '-gs', '--gather_summary_results',
    help="Unpickle all results files and \
        compile into single file. \
        Don't want to do this if already compiled.",
    type=int,
    default=1
)

parser.add_argument(
    '-latex', '--latex_mapping_file',
    help='File path to save tuples which give model \
        string names and latex names.',
    type=str,
    default=None
)
parser.add_argument(
    '-plot_probes', '--probes_plot_file',
    help="File to pickle probes against which to plot expectation values.",
    type=str,
    default=None
)

arguments = parser.parse_args()

directory_to_analyse = arguments.results_directory
log_file = arguments.log_file
all_bayes_csv = arguments.bayes_csv
qhl_mode = bool(arguments.qhl_mode)
further_qhl_mode = bool(arguments.further_qhl_mode)
true_params_path = arguments.true_model_terms_params
exp_data = arguments.use_experimental_data
true_expec_path = arguments.true_expectation_value_path
growth_generator = arguments.growth_generation_rule
gather_summary_results = bool(arguments.gather_summary_results)
true_growth_class = qmla.get_growth_generator_class(
    growth_generation_rule=growth_generator,
    # use_experimental_data=exp_data,
    log_file=log_file
)
dataset = true_growth_class.experimental_dataset
# measurement_type = true_growth_class.measurement_type
latex_mapping_file = arguments.latex_mapping_file
probes_plot_file = arguments.probes_plot_file
# force_plus_probe = bool(arguments.force_plus_probe)
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
    true_model = true_params_info['true_model']
else:
    true_params_dict = None
    true_model = true_growth_class.true_model
true_model_latex = true_growth_class.latex_name(
    true_model
)


if exp_data is False:
    name = true_params_info['true_model']
    terms = database_framework.get_constituent_names_from_name(name)
    params = []
    ops = []
    for t in terms:
        params.append(true_params_dict[t])
        ops.append(database_framework.compute(t))

    true_ham = np.tensordot(params, ops, axes=1)


if not directory_to_analyse.endswith('/'):
    directory_to_analyse += '/'

# Generate results' file names etc depending 
# on what type of QMLA was run
if further_qhl_mode == True:
    results_csv_name = 'summary_further_qhl_results.csv'
    results_csv = directory_to_analyse + results_csv_name
    results_file_name_start = 'further_qhl_results'
    plot_desc = 'further_'
else:
    results_csv_name = 'summary_results.csv'
    results_csv = directory_to_analyse + results_csv_name
    results_file_name_start = 'results'
    plot_desc = ''

# do preliminary analysis 
os.chdir(directory_to_analyse)
pickled_files = []
for file in os.listdir(directory_to_analyse):
    if (
        file.endswith(".p")
        and
        file.startswith(results_file_name_start)
    ):
        pickled_files.append(file)

growth_rules = {}
for f in pickled_files:
    fname = directory_to_analyse + '/' + str(f)
    result = pickle.load(open(fname, 'rb'))
    alph = result['NameAlphabetical']
    if alph not in list(growth_rules.keys()):
        growth_rules[alph] = result['GrowthGenerator']

unique_growth_classes = {}
unique_growth_rules = true_params_info['all_growth_rules']
for g in unique_growth_rules:
    try:
        unique_growth_classes[g] = qmla.get_growth_generator_class(
            growth_generation_rule=g
        )
    except BaseException:
        unique_growth_classes[g] = None

# first get model scores
model_score_results = qmla.analysis.get_model_scores(
    directory_name=directory_to_analyse,
    unique_growth_classes=unique_growth_classes,
    collective_analysis_pickle_file=results_collection_file,
)
model_scores = model_score_results['scores']
growth_rules = model_score_results['growth_rules']
growth_classes = model_score_results['growth_classes']
unique_growth_classes = model_score_results['unique_growth_classes']
median_coeff_determination = model_score_results['avg_coeff_determination']
f_scores = model_score_results['f_scores']
latex_coeff_det = model_score_results['latex_coeff_det']
pickle.dump(
    model_score_results, 
    open(
        os.path.join(
            directory_to_analyse, 
            'champions_info.p'
        ),
        'wb'
    )
)
# rearrange some results... TODO this could be tidier/removed?
models = sorted(model_score_results['wins'].keys())
models.reverse()

f_score = {
    'title': 'F-score',
    'res': [model_score_results['f_scores'][m] for m in models],
    'range': 'cap_1',
}
r_squared = {
    'title': '$R^2$',
    'res': [model_score_results['latex_coeff_det'][m] for m in models],
    'range': 'cap_1',
}

sensitivity = {
    'title': 'Sensitivity',
    'res': [model_score_results['sensitivities'][m] for m in models],
    'range': 'cap_1',
}
precision = {
    'title': 'Precision',
    'res': [model_score_results['precisions'][m] for m in models],
    'range': 'cap_1',
}
wins = {
    'title': '# Wins',
    'res': [model_score_results['wins'][m] for m in models],
    'range': 'uncapped',
}


#######################################
# Now analyse the results.
#######################################

print("\nAnalysing and storing results in", directory_to_analyse)

#######################################
# Gather results
#######################################


if gather_summary_results: 
    # Collect results together into single file. 
    # don't want to waste time doing this if already compiled
    # so gather_summary_results can be set to 0 in analysis script.
    combined_results = qmla.analysis.collect_results_store_csv(
        directory_name = directory_to_analyse,
        results_file_name_start = results_file_name_start,
        results_csv_name = results_csv_name, 
    )

    # Find number of occurences of each model
    # quite costly so it is optional
    plot_num_model_occurences = False 
    if plot_num_model_occurences:
        try:
            qmla.analysis.count_model_occurences(
                latex_map=latex_mapping_file,
                true_model_latex=true_growth_class.latex_name(
                    true_model
                ),
                save_counts_dict=str(
                    directory_to_analyse +
                    "count_model_occurences.p"
                ),
                save_to_file=str(
                    directory_to_analyse +
                    "occurences_of_models.png"
                )
            )
        except BaseException:
            print("ANALYSIS FAILURE: number of occurences for each model.")
            raise
else:
    combined_results = pd.read_csv(
        os.path.join(directory_to_analyse, results_csv_name)
    )

#######################################
# Results/Outputs
## Dynamics
#######################################
try:
    qmla.analysis.plot_dynamics_multiple_models(  # average expected values
        directory_name=directory_to_analyse,
        dataset=dataset,
        results_path=results_csv,
        # use_experimental_data=exp_data,
        results_file_name_start=results_file_name_start,
        true_expectation_value_path=true_expec_path,
        growth_generator=growth_generator,
        unique_growth_classes=unique_growth_classes,
        top_number_models=arguments.top_number_models,
        probes_plot_file=probes_plot_file,
        collective_analysis_pickle_file=results_collection_file,
        save_to_file=str(
            directory_to_analyse +
            plot_desc +
            'expec_vals.png'
        )
    )
except:
    print("ANALYSIS FAILURE: dynamics.")
    raise

#####

#######################################
# Parameter analysis
#######################################
try:
    # Get average parameters of champion models across instances
    average_priors = qmla.analysis.average_parameters_across_instances(
        results_path=results_csv,
        top_number_models=arguments.top_number_models,
        file_to_store = os.path.join(
            directory_to_analyse, 
            'average_priors.p'
        )
    )
except BaseException:
    print("ANALYSIS FAILURE: finding average parameters across instances.")
    raise

try:
    qmla.analysis.average_parameter_estimates(
        directory_name=directory_to_analyse,
        results_path=results_csv,
        top_number_models=arguments.top_number_models,
        results_file_name_start=results_file_name_start,
        growth_generator=growth_generator,
        unique_growth_classes=unique_growth_classes,
        true_params_dict=true_params_dict,
        save_to_file=str(
            directory_to_analyse +
            plot_desc +
            'param_avg.png'
        )
    )
except:
    print("ANALYSIS FAILURE: average parameter plots.")
    raise

# cluster champion learned parameters.
try:
    qmla.analysis.cluster_results_and_plot(
        path_to_results=results_csv,
        true_expec_path=true_expec_path,
        plot_probe_path=probes_plot_file,
        true_params_path=true_params_path,
        growth_generator=growth_generator,
        # measurement_type=measurement_type,
        save_param_values_to_file=str(
            plot_desc + 'clusters_by_param.png'),
        save_param_clusters_to_file=str(
            plot_desc + 'clusters_by_model.png'),
        save_redrawn_expectation_values=str(
            plot_desc + 'clusters_expec_vals.png')
    )
except BaseException:
    print("ANALYSIS FAILURE: clustering plots.")
    raise


#######################################
# QMLA Performance
## model win rates and statistics
#######################################
# model win rates
try:
    qmla.analysis.plot_scores(
        scores=model_scores,
        growth_classes=growth_classes,
        unique_growth_classes=unique_growth_classes,
        growth_rules=growth_rules,
        plot_r_squared=False,
        coefficients_of_determination=median_coeff_determination,
        coefficient_determination_latex_name=latex_coeff_det,
        f_scores=f_scores,
        true_model=true_model,
        growth_generator=growth_generator,
        # collective_analysis_pickle_file = results_collection_file,
        save_file=os.path.join(directory_to_analyse, 'model_wins.png')
    )
except:
    print("ANALYSIS FAILURE: plotting model win rates.")
    raise

# model statistics (f-score, precision, sensitivty)
try:
    qmla.analysis.plot_statistics(
        to_plot = [wins, f_score, precision, sensitivity],
        models = models,
        true_model=true_model_latex,
        save_to_file=str(
            directory_to_analyse +
            'model_stats.png'
        )
    )
except:
    print("ANALYSIS FAILURE: plotting model statistics.")
    raise

try:
    qmla.analysis.count_term_occurences(
        combined_results = combined_results, 
        save_directory = directory_to_analyse
    )
except:
    print("ANALYSIS FAILURE: Counting term occurences.")


# Evaluation: log likelihoods of considered models, compared with champion/true
try:
    qmla.analysis.plot_evaluation_log_likelihoods(
        combined_results = combined_results, 
        save_directory = directory_to_analyse,
        include_median_likelihood=False, 
    )
except: 
    print("ANALYSIS FAILURE: Evaluation log likleihoods.")
    pass

# inspect how nodes perform
try:
    qmla.analysis.inspect_times_on_nodes(
        combined_results = combined_results, 
        save_directory=directory_to_analyse,
    )
except: 
    print("ANALYSIS FAILURE: Time inspection of nodes.")
    pass
    # raise


# model statistics histograms (f-score, precision, sensitivty)
try:
    # Plot metrics such as F1 score histogram
    qmla.analysis.stat_metrics_histograms(
        champ_info = model_score_results, 
        save_to_file=os.path.join(
            directory_to_analyse, 
            'metrics.png'
        )
    )
except: 
    print("ANALYSIS FAILURE: statistical metrics")
    raise

# Summarise results into txt file for quick checking results. 
try:
    qmla.analysis.summarise_qmla_text_file(
        results_csv_path = results_csv, 
        path_to_summary_file = os.path.join(
            directory_to_analyse, 
            'summary.txt'
        )
    )
except:
    print("ANALYSIS FAILURE: summarising txt")
    raise

# Plots used for comparing parameter sweeps; show wins by over/mis/under-fit
try:
    qmla.analysis.parameter_sweep_analysis(
        directory_name=directory_to_analyse,
        results_csv=results_csv,
        save_to_file=os.path.join(
            directory_to_analyse, 
            'sweep_param_total.png'
        )
    )
    qmla.analysis.parameter_sweep_analysis(
        directory_name=directory_to_analyse,
        results_csv=results_csv,
        use_log_times=True,
        use_percentage_models=True,
        save_to_file=os.path.join(
            directory_to_analyse, 
            'sweep_param_percentage.png'
        )
    )
except BaseException:
    print("ANALYSIS FAILURE: parameter sweeps.")
    pass


#######################################
# QMLA Internals
## How QMLA proceeds 
## metrics at each layer
## Quadratic losses, R^2, volume at each experiment
## 
#######################################
try:
    qmla.analysis.generational_analysis(
        combined_results = combined_results, 
        save_directory=directory_to_analyse,
    )
except:
    print("ANALYSIS FAILURE: generational analysis.")
    pass
    # raise

try:
    qmla.analysis.r_sqaured_average(
        results_path=results_csv,
        growth_class=true_growth_class,
        top_number_models=arguments.top_number_models,
        growth_classes_by_name=growth_classes,
        save_to_file=str(
            directory_to_analyse +
            plot_desc +
            'r_squared_averages.png'
        )
    )
except BaseException:
    print(
        "ANALYSIS FAILURE: R^2 against epochs.",
        "R^2 at each epoch not stored in QMLA output (method available in QML)."
    )
    pass

try:
    qmla.analysis.average_quadratic_losses(
        results_path=results_csv,
        growth_classes=unique_growth_classes,
        growth_generator=growth_generator,
        top_number_models=arguments.top_number_models,
        save_to_file=str(
            directory_to_analyse +
            plot_desc +
            'quadratic_losses_avg.png'
        )
    )
except:
    print("ANAYSIS FAILURE: quadratic losses.")
    raise

try:
    qmla.analysis.volume_average(
        results_path=results_csv,
        growth_class=true_growth_class,
        top_number_models=arguments.top_number_models,
        save_to_file=str(
            directory_to_analyse +
            plot_desc +
            'volume_averages.png'
        )
    )
except: 
    print("ANALYSIS FAILURE: volumes.")
    raise

try:
    qmla.analysis.all_times_learned_histogram(
        results_path=results_csv,
        top_number_models=arguments.top_number_models,
        save_to_file=str(
            directory_to_analyse +
            plot_desc +
            'times_learned_upon.png'
        )
    )
except:
    print("ANALYSIS FAILURE: times learned upon.")
    raise

# Bayes factors Vs true model
try:
    qmla.analysis.plot_bayes_factors_v_true_model(
        results_csv_path=all_bayes_csv,
        correct_mod=true_model,
        growth_generator=growth_generator,
        save_to_file=os.path.join(
            directory_to_analyse,
            'bayes_comparisons_true_model.png'
        )
    )
except:
    print("ANALYSIS FAILURE: Bayes factors v true models.")
    pass


#######################################
# Growth rule specific 
## genetic algorithm ananlytics
#######################################

try: 
    qmla.analysis.model_generation_probability(
        # results_path = results_csv,
        combined_results = combined_results,
        save_directory=directory_to_analyse, 
    )
except:
    print("ANALYSIS FAILURE: [gentic algorithm] Model generation rate.")
    pass
    # raise


try:
    qmla.analysis.genetic_alg_fitness_plots(
        results_path = results_csv, 
        save_directory = directory_to_analyse, 
    )
except:
    print("ANALYSIS FAILURE: [genetic algorithm] Fitness measures.")
    # raise
    pass


##################################
# Tree representing all QMLA instances
#######################################
try:
    tree_plot_log = str(directory_to_analyse + 'tree_plot_log.txt')
    sys.stdout = open(
        tree_plot_log, 'w'
    )

    qmla.analysis.plot_tree_multiple_instances(
        results_csv=results_csv,
        latex_mapping_file=latex_mapping_file,
        avg_type='medians',
        all_bayes_csv=all_bayes_csv,
        growth_generator=growth_generator,
        entropy=0,
        inf_gain=0,
        save_to_file='DAG_multi_qmla.png'
    )

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

except ValueError:
    pass
except Exception as exc:
    print("Error plotting multi QMLA tree.")
    print(exc)
    pass

except NameError:
    print(
        "Can not plot multiQMD tree -- this might be because only \
        one instance of QMD was performed. All other plots generated \
        without error."
    )
    pass

except ZeroDivisionError:
    print(
        "Can not plot multiQMD tree -- this might be because only \
        one instance of QMD was performed. All other plots generated \
        without error."
    )
    pass
except BaseException:
    pass