import os
import sys
import numpy as np 
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from scipy import stats
import lfig

__all__ = [
    'plot_dynamics_multiple_models'
]

def format_exponent(n):
    a = '%E' % n
    val = a.split('E')[0].rstrip('0').rstrip('.')
    val = np.round(float(val), 2)
    exponent = a.split('E')[1]

    return str(val) + 'E' + exponent


def plot_dynamics_multiple_models(
    directory_name,
    results_path,
    results_file_name_start='results',
    use_experimental_data=False,
    dataset=None,
    true_expectation_value_path=None,
    probes_plot_file=None,
    exploration_rule=None,
    unique_exploration_classes=None,
    top_number_models=2,
    save_true_expec_vals_alone_plot=True,
    collective_analysis_pickle_file=None,
    return_results=False,
    save_to_file=None
):
    r"""
    Plots reproduced dynamics against time
    for the top models, i.e. those which win the most. 

    TODO: refactor this code - it should not need to unpickle
    all the files which have already been unpickled and stored in the summary
    results CSV.
    TODO: this is a very old method and can surely be improved using Pandas dataframes now stored. 

    :param directory_name: path to directory where results .p files are stored.
    :param results_path: path to CSV with all results for this run.
    :param results_file_name_start: 
    :param use_experimental_data: bool, whether experimental (fixed) data was used.
    :param true_expectation_value_path: path to file containing pre-computed expectation 
        values.
    :param probes_plot_file: path to file with specific probes (states) to use
        for plotting purposes for consistency.  
    :param exploration_rule: the name of the exploration strategy used. 
    :param unique_exploration_classes: dict with single instance of each exploration strategy class
        used in this run.
    :param top_number_models: Number of models to compute averages for 
        (top by number of instance wins). 
    :param true_params_dict: dict with true parameter for each parameter in the 
        true model.
    :param save_true_expec_vals_alone_plot: bool, whether to save a 
        separate plot only of true expectation values, in addition
        to reproduced dynamics.
    :param collective_analysis_pickle_file: if not None, store analysed data
        to this path. 
    :param return_results: bool, to return the analysed data upon function call.
    :param save_to_file: if not None, path to save PNG. 

    :returns None:
    """
    plt.switch_backend('agg')

    # results = pd.DataFrame.from_csv(
    results = pd.read_csv(
        results_path,
        index_col='QID'
    )

    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    def rank_models(n): return sorted(set(n), key=n.count)[::-1]
    # from
    # https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency

    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))

    cm_subsection = np.linspace(0, 0.8, len(winning_models))
    colours = [cm.viridis(x) for x in cm_subsection]

    experimental_measurements = pickle.load(
        open(str(true_expectation_value_path), 'rb')
    )

    expectation_values_by_name = {}
    os.chdir(directory_name)
    pickled_files = []
    for file in os.listdir(directory_name):
        # if file.endswith(".p") and file.startswith("results"):
        if (
            file.endswith(".p")
            and
            file.startswith(results_file_name_start)
        ):
            pickled_files.append(file)
    num_results_files = len(pickled_files)
    exploration_strategies = {}
    for f in pickled_files:
        fname = directory_name + '/' + str(f)
        result = pickle.load(open(fname, 'rb'))
        alph = result['NameAlphabetical']
        expec_values = result['ExpectationValues']

        if alph in expectation_values_by_name.keys():
            expectation_values_by_name[alph].append(expec_values)
        else:
            expectation_values_by_name[alph] = [expec_values]

        if alph not in list(exploration_strategies.keys()):
            exploration_strategies[alph] = result['ExplorationRule']

    exploration_classes = {}
    for g in list(exploration_strategies.keys()):
        try:
            exploration_classes[g] = unique_exploration_classes[exploration_strategies[g]]
        except BaseException:
            exploration_classes[g] = None

    try:
        true_model = unique_exploration_classes[exploration_rule].true_model
    except BaseException:
        print("Couldn't find exploration strategy of {} in \n {}".format(
            exploration_rule,
            unique_exploration_classes
        )
        )
        raise

    collect_expectation_values = {
        'means': {},
        'medians': {},
        'true': {},
        'mean_std_dev': {},
        'success_rate': {},
        'r_squared': {}
    }
    success_rate_by_term = {}
    nmod = len(winning_models)
    if nmod == 1:
        lf = lfig.LatexFigure(auto_label=False, fraction=0.75)
    else:
        ncols = int(np.ceil(np.sqrt(nmod)))
        nrows = int(np.ceil(nmod / ncols)) + 1  # 1 extra row for "master"
        lf = lfig.LatexFigure(gridspec_layout=(nrows, ncols))

    axes_so_far = 1
    full_plot_axis = lf.new_axis(force_position=(0,0), span = (1, 'all'))
    model_statistics = {}

    for term in winning_models:
        expectation_values = {}
        num_sets_of_this_name = len(
            expectation_values_by_name[term]
        )
        for i in range(num_sets_of_this_name):
            learned_expectation_values = (
                expectation_values_by_name[term][i]
            )

            for t in list(experimental_measurements.keys()):
                try:
                    expectation_values[t].append(
                        learned_expectation_values[t]
                    )
                except BaseException:
                    try:
                        expectation_values[t] = [
                            learned_expectation_values[t]
                        ]
                    except BaseException:
                        # if t can't be found, move on
                        pass

        means = {}
        std_dev = {}
        true = {}
        t_values = {}
        lower_iqr_expectation_values = {}
        higher_iqr_expectation_values = {}

        # times = sorted(list(experimental_measurements.keys()))
        true_times = sorted(list(expectation_values.keys()))
        times = sorted(list(expectation_values.keys()))
        times = [
            np.round(t, 2) if t>0.1 else t for t in times
        ]
        flag = True
        one_sample = True
        for t in times:
            means[t] = np.mean(expectation_values[t])
            std_dev[t] = np.std(expectation_values[t])
            lower_iqr_expectation_values[t] = np.percentile(
                expectation_values[t], 25)
            higher_iqr_expectation_values[t] = np.percentile(
                expectation_values[t], 75)
            true[t] = experimental_measurements[t]
            if num_sets_of_this_name > 1:
                expec_values_array = np.array(
                    [[i] for i in expectation_values[t]]
                )
                # print("shape going into ttest:", np.shape(true_expec_values_array))
                if use_experimental_data == True:
                    t_val = stats.ttest_1samp(
                        expec_values_array,  # list of expec vals for this t
                        true[t],  # true expec val of t
                        axis=0,
                        nan_policy='omit'
                    )
                else:
                    true_dist = stats.norm.rvs(
                        loc=true[t],
                        scale=0.001,
                        size=np.shape(expec_values_array)
                    )
                    t_val = stats.ttest_ind(
                        expec_values_array,  # list of expec vals for this t
                        true_dist,  # true expec val of t
                        axis=0,
                        nan_policy='omit'
                    )

                if np.isnan(float(t_val[1])) == False:
                    # t_values[t] = 1-t_val[1]
                    t_values[t] = t_val[1]
                else:
                    print("t_val is nan for t=", t)

        true_exp = [true[t] for t in times]
        # TODO should this be the number of times this model won???
        num_runs = num_sets_of_this_name
        success_rate = 0

        for t in times:

            true_likelihood = true[t]
            mean = means[t]
            std = std_dev[t]
            credible_region = (2 / np.sqrt(num_runs)) * std

            if (
                (true_likelihood < (mean + credible_region))
                and
                (true_likelihood > (mean - credible_region))
            ):
                success_rate += 1 / len(times)

        mean_exp = np.array([means[t] for t in times])
        std_dev_exp = np.array([std_dev[t] for t in times])
        lower_iqr_exp = np.array(
            [lower_iqr_expectation_values[t] for t in times])
        higher_iqr_exp = np.array(
            [higher_iqr_expectation_values[t] for t in times])
        residuals = (mean_exp - true_exp)**2
        sum_residuals = np.sum(residuals)
        mean_true_val = np.mean(true_exp)
        true_mean_minus_val = (true_exp - mean_true_val)**2
        sum_of_squares = np.sum(
            true_mean_minus_val
        )
        if sum_of_squares != 0:
            final_r_squared = 1 - sum_residuals / sum_of_squares
        else:
            print("[multiQMD plots] sum of squares 0")
            final_r_squared = -100

        # R^2 for interquartile range
        lower_iqr_sum_residuals = np.sum(
            (lower_iqr_exp - true_exp)**2
        )
        lower_iqr_sum_of_squares = np.sum(
            (lower_iqr_exp - np.mean(lower_iqr_exp))**2
        )
        lower_iqr_r_sq = 1 - (lower_iqr_sum_residuals /
                              lower_iqr_sum_of_squares)
        higher_iqr_sum_residuals = np.sum(
            (higher_iqr_exp - true_exp)**2
        )
        higher_iqr_sum_of_squares = np.sum(
            (higher_iqr_exp - np.mean(higher_iqr_exp))**2
        )
        higher_iqr_r_sq = 1 - (higher_iqr_sum_residuals /
                               higher_iqr_sum_of_squares)

        name = exploration_classes[term].latex_name(term)
        try:
            description = r" {} [R^2={}]".format(name, format_exponent(final_r_squared))
        except BaseException:
            description = r" {} [R^2=NaN]".format(name)

        if term == true_model:
            description += " (True)"

        description_w_bayes_t_value = str(
            name + ' : ' +
            str(round(success_rate, 2)) +
            ' (' + str(num_sets_of_this_name) + ').'
        )

        collect_expectation_values['means'][name] = mean_exp
        collect_expectation_values['mean_std_dev'][name] = std_dev_exp
        collect_expectation_values['success_rate'][name] = success_rate
        model_statistics[name] = {
            'r_squared_median_exp_val': final_r_squared,
            'mean_expectation_values': mean_exp,
            'mean_std_dev': std_dev_exp,
            'success_rate_t_test': success_rate,
            'num_wins': num_sets_of_this_name,
            'win_percentage': int(100 * num_sets_of_this_name / num_results_files),
            'num_instances': num_results_files,
            'lower_iqr_exp_val': lower_iqr_exp,
            'higher_iqr_exp_val': higher_iqr_exp,
            'lower_iqr_r_sq': lower_iqr_r_sq,
            'higher_iqr_r_sq': higher_iqr_r_sq,
            'times': times
        }
        if nmod > 1:
            ax = lf.new_axis()
            ax.plot(
                times,
                mean_exp,
                c=colours[winning_models.index(term)],
                label=description
            )
            ax.fill_between(
                times,
                mean_exp - std_dev_exp,
                mean_exp + std_dev_exp,
                alpha=0.2,
                facecolor=colours[winning_models.index(term)],
            )
            ax.set_ylim(0, 1)
            ax.set_xlim(0, max(times))

            success_rate_by_term[term] = success_rate
            ax.set_title('Mean Expectation Values')
            ax.scatter(
                times,
                true_exp,
                color='r',
                s=5,
                label='System'
            )
            ax.plot(
                times,
                true_exp,
                color='r',
                alpha=0.3
            )
            ax.set_title(description)

        # Add this model to "master" plot

        high_level_label = str(name)
        if term == true_model:
            high_level_label += ' (True)'

        full_plot_axis.plot(
            times,
            mean_exp,
            c=colours[winning_models.index(term)],
            label=high_level_label
        )
        full_plot_axis.scatter(
            times,
            true_exp,
            color='r',
            s=5,
            label='System'
        )
        full_plot_axis.plot(
            times,
            true_exp,
            color='r',
            alpha=0.3
        )

    full_plot_axis.legend()
    full_plot_axis.set_ylim(0, 1)
    full_plot_axis.set_xlim(0, max(times))
    if nmod > 1:
        lf.fig.text(0.45, -0.04, 'Time', ha='center')
        lf.fig.text(-0.04, 0.5, 'Expectation Value', va='center', rotation='vertical')
    else:
        full_plot_axis.set_ylabel("Expectation value")
        full_plot_axis.set_xlabel("Time (a.u)")

    if save_to_file is not None:
        lf.fig.suptitle("Dynamics of trained models.")
        lf.save(save_to_file)

    # Also save an image of the only the system dynamics
    if (
        save_true_expec_vals_alone_plot == True
        and
        save_to_file is not None
    ):
        lf = lfig.LatexFigure(fraction=0.75, auto_label=False)
        ax = lf.new_axis()
        ax.plot(
            times,
            true_exp,
            marker='o',
            color='r',
            label='True System'
            # alpha = 0.3
        )
        ax.set_xlabel('Time')
        ax.set_ylabel('Expectation Value')
        ax.legend()
        true_only_fig_file = str(
            save_to_file[:-4]
            + '_true_expec_vals.png'
        )
        ax.set_title("Expectation Values of True model.")
        lf.save(
            true_only_fig_file,
        )

    # add the combined analysis dict
    collect_expectation_values['times'] = true_times
    collect_expectation_values['true'] = true_exp

    if collective_analysis_pickle_file is not None:
        if os.path.isfile(collective_analysis_pickle_file) is False:
            pickle.dump(
                model_statistics,
                open(collective_analysis_pickle_file, 'wb')
            )
        else:
            # load current analysis dict, add to it and rewrite it.
            combined_analysis = pickle.load(
                open(
                    collective_analysis_pickle_file,
                    'rb'
                )
            )
            for model in model_statistics.keys():
                new_keys = list(model_statistics[model].keys())
                for key in new_keys:
                    combined_analysis[model][key] = model_statistics[model][key]
            pickle.dump(
                combined_analysis,
                open(collective_analysis_pickle_file, 'wb')
            )
    else:
        print(
            "[analyse] collective analysis path:",
            collective_analysis_pickle_file)

    if return_results == True:
        expectation_values_by_latex_name = {}
        for term in winning_models:
            latex_name = unique_exploration_classes[exploration_rule].latex_name(
                term)
            expectation_values_by_latex_name[latex_name] = expectation_values_by_name[term]

        return times, mean_exp, std_dev_exp, winning_models, term, true, description, expectation_values_by_latex_name, expectation_values_by_name

