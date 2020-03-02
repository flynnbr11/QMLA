import numpy as np
import argparse
from matplotlib.lines import Line2D
import sys
import os
import pickle
import pandas

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


from qmla.analysis.analysis_and_plot_functions import fill_between_sigmas, cumulativeQMDTreePlot
import qmla.get_growth_rule as get_growth_rule
import qmla.model_naming as model_naming
# import qmla.PlotQMD as ptq
import qmla.database_framework as database_framework
plt.switch_backend('agg')

def flatten(l): 
    # flatten list of lists
    return [item for sublist in l for item in sublist]

def parameter_sweep_analysis(
    directory_name,
    results_csv,
    save_to_file=None,
    use_log_times=False,
    use_percentage_models=False
):

    import os
    import csv
    if not directory_name.endswith('/'):
        directory_name += '/'

    # qmd_cumulative_results = pandas.DataFrame.from_csv(results_csv,
    qmd_cumulative_results = pandas.read_csv(results_csv,
                                                       index_col='ConfigLatex'
                                                       )
    piv = pandas.pivot_table(
        qmd_cumulative_results,
        values=['CorrectModel', 'Time', 'Overfit', 'Underfit', 'Misfit'],
        index=['ConfigLatex'],
        aggfunc={
            'Time': [np.mean, np.median, min, max],
            'CorrectModel': [np.sum, np.mean],
            'Overfit': [np.sum, np.mean],
            'Misfit': [np.sum, np.mean],
            'Underfit': [np.sum, np.mean]
        }
    )

    time_means = list(piv['Time']['mean'])
    time_mins = list(piv['Time']['min'])
    time_maxs = list(piv['Time']['max'])
    time_medians = list(piv['Time']['median'])
    correct_count = list(piv['CorrectModel']['sum'])
    correct_ratio = list(piv['CorrectModel']['mean'])
    overfit_count = list(piv['Overfit']['sum'])
    overfit_ratio = list(piv['Overfit']['mean'])
    underfit_count = list(piv['Underfit']['sum'])
    underfit_ratio = list(piv['Underfit']['mean'])
    misfit_count = list(piv['Misfit']['sum'])
    misfit_ratio = list(piv['Misfit']['mean'])
    num_models = len(time_medians)

    configs = piv.index.tolist()
    percentages = [a * 100 for a in correct_ratio]

    plt.clf()
    fig, ax = plt.subplots()
    if num_models <= 5:
        plot_height = num_models
    else:
        plot_height = num_models / 2

    fig.set_figheight(plot_height)
    # fig.set_figwidth(num_models/4)

    ax2 = ax.twiny()
    width = 0.5  # the width of the bars
    ind = np.arange(len(correct_ratio))  # the x locations for the groups

    if use_log_times:
        times_to_use = [np.log10(t) for t in time_medians]
        ax2.set_xlabel('Time ($log_{10}$ seconds)')
    else:
        times_to_use = time_medians
        ax2.set_xlabel('Median Time (seconds)')

    if use_percentage_models:
        correct = [a * 100 for a in correct_ratio]
        misfit = [a * 100 for a in misfit_ratio]
        underfit = [a * 100 for a in underfit_ratio]
        overfit = [a * 100 for a in overfit_ratio]
        ax.set_xlabel('% Models')
    else:
        correct = correct_count
        misfit = misfit_count
        overfit = overfit_count
        underfit = underfit_count
        ax.set_xlabel('Number of Models')

    max_x = correct[0] + misfit[0] + overfit[0] + underfit[0]
    time_colour = 'b'
    ax2.barh(ind, times_to_use, width / 4, color=time_colour, label='Time')

    times_to_mark = [60, 600, 3600, 14400, 36000]
    if use_log_times:
        times_to_mark = [np.log10(t) for t in times_to_mark]

    max_time = max(times_to_use)
    for t in times_to_mark:
        if t < max_time:
            ax2.axvline(x=t, color=time_colour)

    left_pts = [0] * num_models
    ax.barh(ind, correct, width, color='g', align='center',
            label='Correct Models', left=left_pts
            )
    left_pts = [sum(x) for x in zip(left_pts, correct)]

    ax.barh(ind, underfit, width, color='r', align='center',
            label='Underfit Models', left=left_pts
            )
    left_pts = [sum(x) for x in zip(left_pts, underfit)]

    ax.barh(ind, misfit, width, color='orange', align='center',
            label='Misfit Models', left=left_pts
            )
    left_pts = [sum(x) for x in zip(left_pts, misfit)]

    ax.barh(ind, overfit, width, color='y', align='center',
            label='Overfit Models', left=left_pts
            )
    left_pts = [sum(x) for x in zip(left_pts, overfit)]

#    ax.axvline(x=max_x/2, color='g', label='50% Models correct')
    ax.set_yticks(ind)
    ax.set_yticklabels(configs, minor=False)
    ax.set_ylabel('Configurations')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center',
               bbox_to_anchor=(0.5, -0.2), ncol=2
               )

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def average_parameters(
    results_path,
    top_number_models=3,
    average_type='median'
):

    # results = pandas.DataFrame.from_csv(
    results = pandas.read_csv(
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

    params_dict = {}
    sigmas_dict = {}
    for mod in winning_models:
        params_dict[mod] = {}
        sigmas_dict[mod] = {}
        params = database_framework.get_constituent_names_from_name(mod)
        for p in params:
            params_dict[mod][p] = []
            sigmas_dict[mod][p] = []

    for i in range(len(winning_models)):
        mod = winning_models[i]
        learned_parameters = list(
            results[results['NameAlphabetical'] == mod]
            ['LearnedParameters']
        )
        final_sigmas = list(
            results[results['NameAlphabetical'] == mod]
            ['FinalSigmas']
        )
        num_wins_for_mod = len(learned_parameters)
        for i in range(num_wins_for_mod):
            params = eval(learned_parameters[i])
            sigmas = eval(final_sigmas[i])
            for k in list(params.keys()):
                params_dict[mod][k].append(params[k])
                sigmas_dict[mod][k].append(sigmas[k])

    average_params_dict = {}
    avg_sigmas_dict = {}
    std_deviations = {}
    learned_priors = {}
    for mod in winning_models:
        average_params_dict[mod] = {}
        avg_sigmas_dict[mod] = {}
        std_deviations[mod] = {}
        learned_priors[mod] = {}
        params = database_framework.get_constituent_names_from_name(mod)
        for p in params:
            # if average_type == 'median':
            #     average_params_dict[mod][p] = np.median(
            #         params_dict[mod][p]
            #     )
            # else:
            #     average_params_dict[mod][p] = np.mean(
            #         params_dict[mod][p]
            #     )
            # if np.std(params_dict[mod][p]) > 0:
            #     std_deviations[mod][p] = np.std(params_dict[mod][p])
            # else:
            #     # if only one winner, give relatively broad prior.
            #     std_deviations[mod][p] = 0.5

            # learned_priors[mod][p] = [
            #     average_params_dict[mod][p],
            #     std_deviations[mod][p]
            # ]

            avg_sigmas_dict[mod][p] = np.median(sigmas_dict[mod][p])
            averaging_weight = [1 / sig for sig in sigmas_dict[mod][p]]
            # print("[mod][p]:", mod, p)
            # print("Attempting to avg this list:", params_dict[mod][p])
            # print("with these weights:", averaging_weight)

            average_params_dict[mod][p] = np.average(
                params_dict[mod][p],
                weights=sigmas_dict[mod][p]
            )
            # print("avg sigmas dict type:", type(avg_sigmas_dict[mod][p]))
            # print("type average_params_dict:", type(average_params_dict[mod][p]))
            # print("avg sigmas dict[mod][p]:", avg_sigmas_dict[mod][p])
            # print("average_params_dict[mod][p]:", average_params_dict[mod][p])
            learned_priors[mod][p] = [
                average_params_dict[mod][p],
                avg_sigmas_dict[mod][p]
            ]

    print("Average Parmaeters plot complete")
    return learned_priors


def average_parameter_estimates(
    directory_name,
    results_path,
    results_file_name_start='results',
    growth_generator=None,
    unique_growth_classes=None,
    top_number_models=2,
    true_params_dict=None,
    save_to_file=None
):
    from matplotlib import cm
    plt.switch_backend('agg')  # to try fix plt issue on BC
    # results = pandas.DataFrame.from_csv(
    results = pandas.read_csv(
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

    os.chdir(directory_name)
    pickled_files = []
    for file in os.listdir(directory_name):
        if file.endswith(".p") and file.startswith(results_file_name_start):
            pickled_files.append(file)

    parameter_estimates_from_qmd = {}
    num_experiments_by_name = {}

    latex_terms = {}
    growth_rules = {}

    for f in pickled_files:
        fname = directory_name + '/' + str(f)
        result = pickle.load(open(fname, 'rb'))
        alph = result['NameAlphabetical']
        track_parameter_estimates = result['Trackplot_parameter_estimates']

        # num_experiments = result['NumExperiments']
        if alph in parameter_estimates_from_qmd.keys():
            parameter_estimates_from_qmd[alph].append(
                track_parameter_estimates)
        else:
            parameter_estimates_from_qmd[alph] = [track_parameter_estimates]
            num_experiments_by_name[alph] = result['NumExperiments']

        if alph not in list(growth_rules.keys()):
            try:
                growth_rules[alph] = result['GrowthGenerator']
            except BaseException:
                growth_rules[alph] = growth_generator

    unique_growth_rules = list(set(list(growth_rules.values())))
    growth_classes = {}
    for g in list(growth_rules.keys()):
        try:
            growth_classes[g] = unique_growth_classes[growth_rules[g]]
        except BaseException:
            growth_classes[g] = None
    # print("[AnalyseMultiple - param avg] unique_growth_rules:", unique_growth_rules)
    # print("[AnalyseMultiple - param avg] unique_growth_classes:", unique_growth_classes)
    # print("[AnalyseMultiple - param avg] growth classes:", growth_classes)

    for name in winning_models:
        num_experiments = num_experiments_by_name[name]
        # epochs = range(1, 1+num_experiments)
        epochs = range(num_experiments_by_name[name] + 1)

        plt.clf()
        fig = plt.figure()
        ax = plt.subplot(111)

        parameters_for_this_name = parameter_estimates_from_qmd[name]
        num_wins_for_name = len(parameters_for_this_name)
        terms = sorted(database_framework.get_constituent_names_from_name(name))
        num_terms = len(terms)

        ncols = int(np.ceil(np.sqrt(num_terms)))
        nrows = int(np.ceil(num_terms / ncols))

        fig, axes = plt.subplots(
            figsize=(10, 7),
            nrows=nrows,
            ncols=ncols,
            squeeze=False,
        )
        row = 0
        col = 0
        axes_so_far = 0

        cm_subsection = np.linspace(0, 0.8, num_terms)
#        colours = [ cm.magma(x) for x in cm_subsection ]
        colours = [cm.Paired(x) for x in cm_subsection]

        parameters = {}

        for t in terms:
            parameters[t] = {}

            for e in epochs:
                parameters[t][e] = []

        for i in range(len(parameters_for_this_name)):
            track_params = parameters_for_this_name[i]
            for t in terms:
                for e in epochs:
                    parameters[t][e].append(track_params[t][e])

        avg_parameters = {}
        std_devs = {}
        for p in terms:
            avg_parameters[p] = {}
            std_devs[p] = {}

            for e in epochs:
                avg_parameters[p][e] = np.median(parameters[p][e])
                std_devs[p][e] = np.std(parameters[p][e])

        for term in sorted(terms):
            ax = axes[row, col]
            axes_so_far += 1
            col += 1
            if (row == 0 and col == ncols):
                leg = True
            else:
                leg = False

            if col == ncols:
                col = 0
                row += 1
            # latex_terms[term] = database_framework.latex_name_ising(term)
            latex_terms[term] = growth_classes[name].latex_name(term)
            averages = np.array(
                [avg_parameters[term][e] for e in epochs]
            )
            standard_dev = np.array(
                [std_devs[term][e] for e in epochs]
            )

            try:
                true_val = true_params_dict[term]
                # true_term_latex = database_framework.latex_name_ising(term)
                true_term_latex = growth_classes[name].latex_name(term)
                ax.axhline(
                    true_val,
                    # label=str(true_term_latex+ ' True'),
                    # color=colours[terms.index(term)]
                    label=str('True value'),
                    color='black'

                )
            except BaseException:
                pass

            ax.axhline(
                0,
                linestyle='--',
                alpha=0.5,
                color='black',
                label='0'
            )
            fill_between_sigmas(
                ax,
                parameters[term],
                # [e +1 for e in epochs],
                epochs,
                legend=leg
            )

            ax.scatter(
                [e + 1 for e in epochs],
                #                epochs,
                averages,
                s=max(1, 50 / num_experiments),
                label=latex_terms[term],
                # color=colours[terms.index(term)]
                color='black'
            )

            # latex_term = database_framework.latex_name_ising(term)
            latex_term = growth_classes[name].latex_name(term)
            # latex_term = latex_terms[term]
            ax.set_title(str(latex_term))

        """
        plot_title= str(
            'Average Parameter Estimates '+
            # str(database_framework.latex_name_ising(name)) +
            ' [' +
            str(num_wins_for_name) + # TODO - num times this model won
            ' instances].'
        )
        ax.set_ylabel('Parameter Esimate')
        ax.set_xlabel('Experiment')
        plt.title(plot_title)
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title='Parameter'
        )
        """

        latex_name = growth_classes[name].latex_name(term)

        if save_to_file is not None:
            fig.suptitle(
                'Parameter Esimates for {}'.format(latex_name)
            )
            try:
                save_file = ''
                if save_to_file[-4:] == '.png':
                    partial_name = save_to_file[:-4]
                    save_file = str(partial_name + '_' + name + '.png')
                else:
                    save_file = str(save_to_file + '_' + name + '.png')
                # plt.tight_layout()
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(save_file, bbox_inches='tight')
            except BaseException:
                print("Filename too long. Defaulting to idx")
                save_file = ''
                if save_to_file[-4:] == '.png':
                    partial_name = save_to_file[:-4]
                    save_file = str(
                        partial_name +
                        '_' +
                        str(winning_models.index(name)) +
                        '.png'
                    )
                else:
                    save_file = str(save_to_file + '_' + name + '.png')
                # plt.tight_layout()
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(save_file, bbox_inches='tight')


def analyse_and_plot_dynamics_multiple_models(
    directory_name,
    dataset,
    results_path,
    results_file_name_start='results',
    use_experimental_data=False,
    true_expectation_value_path=None,
    growth_generator=None,
    unique_growth_classes=None,
    probes_plot_file=None,
    top_number_models=2,
    save_true_expec_vals_alone_plot=True,
    collective_analysis_pickle_file=None,
    return_results=False,
    save_to_file=None
):
    print("[Bayes t test] unique_growth_classes:", unique_growth_classes)

    plt.switch_backend('agg')
    from matplotlib import cm
    from scipy import stats

    # results = pandas.DataFrame.from_csv(
    results = pandas.read_csv(
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
    # colours = [ cm.Spectral(x) for x in cm_subsection ]

    # Relies on Results folder structure -- not safe?!
    # ie ExperimentalSimulations/Results/Sep_10/14_32/results_001.p, etc
    if use_experimental_data == True:
        os.chdir(directory_name)
        # os.chdir("../../../../ExperimentalSimulations/Data/")
        os.chdir("../../../../Launch/Data/")
        experimental_measurements = pickle.load(
            open(str(dataset), 'rb')
        )
    elif true_expectation_value_path is not None:
        experimental_measurements = pickle.load(
            open(str(true_expectation_value_path), 'rb')
        )
    else:
        print("Either set \
            use_experimental_data=True or \
            provide true_expectation_value_path"
              )
        return False

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
    growth_rules = {}
    for f in pickled_files:
        fname = directory_name + '/' + str(f)
        result = pickle.load(open(fname, 'rb'))
        alph = result['NameAlphabetical']
        expec_values = result['ExpectationValues']

        if alph in expectation_values_by_name.keys():
            expectation_values_by_name[alph].append(expec_values)
        else:
            expectation_values_by_name[alph] = [expec_values]

        if alph not in list(growth_rules.keys()):
            growth_rules[alph] = result['GrowthGenerator']

    growth_classes = {}
    for g in list(growth_rules.keys()):
        try:
            growth_classes[g] = unique_growth_classes[growth_rules[g]]
        except BaseException:
            growth_classes[g] = None

    try:
        true_model = unique_growth_classes[growth_generator].true_model
    except BaseException:
        print("Couldn't find growth rule of {} in \n {}".format(
            growth_generator,
            unique_growth_classes
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
    ncols = int(np.ceil(np.sqrt(nmod)))
    nrows = int(np.ceil(nmod / ncols)) + 1  # 1 extra row for "master"

    fig = plt.figure(
        figsize=(15, 8),
        # constrained_layout=True,
        tight_layout=True
    )
    gs = GridSpec(
        nrows,
        ncols,
        # figure=fig # not available on matplotlib 2.1.1 (on BC)
    )

    row = 1
    col = 0

    axes_so_far = 1
    # full_plot_axis = axes[0,0]
    full_plot_axis = fig.add_subplot(gs[0, :])
    # i=0
    model_statistics = {}

    for term in winning_models:
        # plt.clf()
        # ax.clf()
        # ax = axes[row, col]
        ax = fig.add_subplot(gs[row, col])
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
        times = [np.round(t, 2) for t in times]
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
        # name=database_framework.latex_name_ising(term)
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

        name = growth_classes[term].latex_name(term)
        try:
            description = str(
                name +
                ' (' + str(num_sets_of_this_name) + ')'
                + ' [$R^2=$' +
                str(
                    # np.round(final_r_squared, 2)
                    # np.format_float_scientific(
                    #     final_r_squared,
                    #     precision=2
                    # )
                    format_exponent(final_r_squared)
                )
                + ']'
            )
        except BaseException:
            print(
                "Failed to format exponent; final r squared:",
                final_r_squared)
            description = str(
                name +
                ' (' + str(num_sets_of_this_name) + ')'
                + ' [$R^2=0$]'
            )

        if term == true_model:
            description += ' (True)'

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

        # plt.title('Mean Expectation Values')
        # plt.xlabel('Time')
        # plt.ylabel('Expectation Value')
        # true_exp = [true[t] for t in times]
        # ax.set_xlim(0,1)
        # plt.xlim(0,1)

        ax.set_title('Mean Expectation Values')
        # if col == 0:
        #     ax.set_ylabel('Expectation Value')
        # if row == nrows-1:
        #     ax.set_xlabel('Time')
        # ax.set_xlim(0,1)
        # plt.xlim(0,1)

        ax.scatter(
            times,
            true_exp,
            color='r',
            s=5,
            label='True Expectation Value'
        )
        ax.plot(
            times,
            true_exp,
            color='r',
            alpha=0.3
        )

        # ax.legend(
        #     loc='center left',
        #     bbox_to_anchor=(1, 0.5),
        #     title=' Model : Bayes t-test (instances)'
        # )

        # fill in "master" plot

        high_level_label = str(name)
        if term == true_model:
            high_level_label += ' (True)'

        full_plot_axis.plot(
            times,
            mean_exp,
            c=colours[winning_models.index(term)],
            label=high_level_label
        )
        if axes_so_far == 1:
            full_plot_axis.scatter(
                times,
                true_exp,
                color='r',
                s=5,
                label='True Expectation Value'
            )
            full_plot_axis.plot(
                times,
                true_exp,
                color='r',
                alpha=0.3
            )
        full_plot_axis.legend(
            loc='center left',
            bbox_to_anchor=(1, 0),
        )
        # full_plot_axis.legend(
        #     ncol = ncols,
        #     loc='lower center',
        #     bbox_to_anchor=(0.5, -1.3),
        # )
        full_plot_axis.set_ylim(0, 1)
        full_plot_axis.set_xlim(0, max(times))

        axes_so_far += 1
        col += 1
        if col == ncols:
            col = 0
            row += 1
        # ax.set_title(str(name))
        ax.set_title(description)

    fig.text(0.45, -0.04, 'Time', ha='center')
    fig.text(-0.04, 0.5, 'Expectation Value', va='center', rotation='vertical')

    if save_to_file is not None:
        fig.suptitle("Expectation Values of learned models.")
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_to_file, bbox_inches='tight')

    # Also save an image of the true expectation values without overlaying
    # results
    if (
        save_true_expec_vals_alone_plot == True
        and
        save_to_file is not None
    ):
        plt.clf()
        plt.plot(
            times,
            true_exp,
            marker='o',
            color='r',
            label='True System'
            # alpha = 0.3
        )
        plt.xlabel('Time')
        plt.ylabel('Expectation Value')
        plt.legend()
        true_only_fig_file = str(
            save_to_file[:-4]
            + '_true_expec_vals.png'
        )
        plt.title("Expectation Values of True model.")
        plt.savefig(
            true_only_fig_file,
            bbox_inches='tight'
        )

    # add the combined analysis dict
    collect_expectation_values['times'] = true_times
    collect_expectation_values['true'] = true_exp

    if collective_analysis_pickle_file is not None:
        if os.path.isfile(collective_analysis_pickle_file) is False:
            # combined_analysis = {
            #     # 'expectation_values' : collect_expectation_values,
            #     # 'statistics' : model_statistics,
            #     'model_statistics' : model_statistics,
            # }
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
            latex_name = unique_growth_classes[growth_generator].latex_name(
                term)
            expectation_values_by_latex_name[latex_name] = expectation_values_by_name[term]

        return times, mean_exp, std_dev_exp, winning_models, term, true, description, expectation_values_by_latex_name, expectation_values_by_name


def format_exponent(n):
    a = '%E' % n
    val = a.split('E')[0].rstrip('0').rstrip('.')
    val = np.round(float(val), 2)
    exponent = a.split('E')[1]

    return str(val) + 'E' + exponent


def r_sqaured_average(
    results_path,
    growth_class,
    growth_classes_by_name,
    top_number_models=2,
    save_to_file=None
):
    from matplotlib import cm
    plt.clf()
    fig = plt.figure()
    ax = plt.subplot(111)

    # results = pandas.DataFrame.from_csv(
    results = pandas.read_csv(
        results_path,
        index_col='QID'
    )
    all_winning_models = list(results.loc[:, 'NameAlphabetical'])
    def rank_models(n): return sorted(set(n), key=n.count)[::-1]
    # from
    # https://codegolf.stackexchange.com/questions/17287/sort-the-distinct-elements-of-a-list-in-descending-order-by-frequency

    r_sq_by_model = {}

    if len(all_winning_models) > top_number_models:
        winning_models = rank_models(all_winning_models)[0:top_number_models]
    else:
        winning_models = list(set(all_winning_models))

    names = winning_models
    num_models = len(names)
    cm_subsection = np.linspace(0, 0.8, num_models)
    #        colours = [ cm.magma(x) for x in cm_subsection ]
    colours = [cm.viridis(x) for x in cm_subsection]

    i = 0
    for i in range(len(names)):
        name = names[i]
        r_squared_values = list(
            results[results['NameAlphabetical'] == name]['RSquaredByEpoch']
        )

        r_squared_lists = {}
        num_wins = len(r_squared_values)
        for j in range(num_wins):
            rs = eval(r_squared_values[j])
            for t in list(rs.keys()):
                try:
                    r_squared_lists[t].append(rs[t])
                except BaseException:
                    r_squared_lists[t] = [rs[t]]

        times = sorted(list(r_squared_lists.keys()))
        means = np.array(
            [np.mean(r_squared_lists[t]) for t in times]
        )
        std_dev = np.array(
            [np.std(r_squared_lists[t]) for t in times]
        )

        # term = database_framework.latex_name_ising(name)
        gr_class = growth_classes_by_name[name]
        # TODO need growth rule of given name to get proper latex term
        term = gr_class.latex_name(name)
        # term = growth_class.latex_name(name) # TODO need growth rule of given
        # name to get proper latex term
        r_sq_by_model[term] = means
        plot_label = str(term + ' (' + str(num_wins) + ')')
        colour = colours[i]
        ax.plot(
            times,
            means,
            label=plot_label,
            marker='o'
        )
        ax.fill_between(
            times,
            means - std_dev,
            means + std_dev,
            alpha=0.2
        )
        ax.legend(
            bbox_to_anchor=(1.0, 0.9),
            title='Model (# instances)'
        )
    print("[AnalyseMultiple - r sq] r_sq_by_model:", r_sq_by_model)

    plt.xlabel('Epoch')
    plt.ylabel('$R^2$')
    plt.title('$R^2$ average')

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def volume_average(
    results_path,
    growth_class,
    top_number_models=2,
    save_to_file=None
):
    from matplotlib import cm
    plt.clf()
    fig = plt.figure()
    ax = plt.subplot(111)

    # results = pandas.DataFrame.from_csv(
    results = pandas.read_csv(
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

    names = winning_models
    num_models = len(names)
    cm_subsection = np.linspace(0, 0.8, num_models)
    #        colours = [ cm.magma(x) for x in cm_subsection ]
    colours = [cm.viridis(x) for x in cm_subsection]

    i = 0
    for i in range(len(names)):
        name = names[i]
        volume_values = list(
            results[results['NameAlphabetical'] == name]['TrackVolume']
        )

        volume_lists = {}
        num_wins = len(volume_values)
        for j in range(num_wins):
            rs = eval(volume_values[j])
            for t in list(rs.keys()):
                try:
                    volume_lists[t].append(rs[t])
                except BaseException:
                    volume_lists[t] = [rs[t]]

        times = sorted(list(volume_lists.keys()))
        means = np.array(
            [np.mean(volume_lists[t]) for t in times]
        )

        std_dev = np.array(
            [np.std(volume_lists[t]) for t in times]
        )

        # term = database_framework.latex_name_ising(name)
        term = growth_class.latex_name(name)
        plot_label = str(term + ' (' + str(num_wins) + ')')
        colour = colours[i]
        ax.plot(
            times,
            means,
            label=plot_label,
            marker='o',
            markevery=10
        )
        ax.fill_between(
            times,
            means - std_dev,
            means + std_dev,
            alpha=0.2
        )
        ax.legend(
            bbox_to_anchor=(1.0, 0.9),
            title='Model (# instances)'
        )
    plt.semilogy()
    plt.xlabel('Epoch')
    plt.ylabel('Volume')
    plt.title('Volume average')

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def all_times_learned_histogram(
    results_path="summary_results.csv",
    top_number_models=2,
    save_to_file=None
):
    from matplotlib import cm
    plt.clf()
    fig = plt.figure()
    ax = plt.subplot(111)

    # results = pandas.DataFrame.from_csv(
    results = pandas.read_csv(
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

    names = winning_models
    num_models = len(names)
    cm_subsection = np.linspace(0, 0.8, num_models)
    #        colours = [ cm.magma(x) for x in cm_subsection ]
    colours = [cm.viridis(x) for x in cm_subsection]

    times_by_model = {}
    max_time = 0
    for i in range(len(names)):
        name = names[i]
        model_colour = colours[i]
        times_by_model[name] = []
        this_model_times_separate_runs = list(
            results[results['NameAlphabetical'] == name]['TrackTimesLearned']
        )

        num_wins = len(this_model_times_separate_runs)
        for j in range(num_wins):
            this_run_times = eval(this_model_times_separate_runs[j])
            times_by_model[name].extend(this_run_times)
            if max(this_run_times) > max_time:
                max_time = max(this_run_times)
        times_this_model = times_by_model[name]
        model_label = str(
            list(results[results['NameAlphabetical'] == name]['ChampLatex'])[0]
        )

        plt.hist(
            times_this_model,
            color=model_colour,
            # histtype='stepfilled',
            histtype='step',
            # histtype='bar',
            fill=False,
            label=model_label
        )

    # presuming all models used same heuristics .... TODO change if models can
    # use ones
    heuristic_type = list(
        results[results['NameAlphabetical'] == names[0]]['Heuristic'])[0]

    plt.legend()
    plt.title("Times learned on [{}]".format(heuristic_type))
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()
    plt.semilogy()
    if max_time > 100:
        plt.semilogx()
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


def get_model_scores(
    directory_name,
    unique_growth_classes,
    collective_analysis_pickle_file=None,
):

    os.chdir(directory_name)

    scores = {}
    growth_rules = {}

    pickled_files = []
    for file in os.listdir(directory_name):
        if file.endswith(".p") and file.startswith("results"):
            pickled_files.append(file)

    coeff_of_determination = {}
    avg_coeff_determination = {}
    f_scores = {}
    precisions = {}
    sensitivities = {}
    volumes = {}
    model_results = {}

    for f in pickled_files:
        fname = directory_name + '/' + str(f)
        result = pickle.load(open(fname, 'rb'))
        alph = result['NameAlphabetical']
        vol = result['TrackVolume']
        vol_list = [vol[e] for e in list(sorted(result['TrackVolume'].keys()))]

        if alph in scores.keys():
            scores[alph] += 1
            coeff_of_determination[alph].append(result['FinalRSquared'])
            volumes[alph].append(vol_list)

        else:
            scores[alph] = 1
            coeff_of_determination[alph] = [result['FinalRSquared']]
            f_scores[alph] = result['Fscore']
            sensitivities[alph] = result['Sensitivity']
            precisions[alph] = result['Precision']
            volumes[alph] = [vol_list]

        if alph not in list(growth_rules.keys()):
            growth_rules[alph] = result['GrowthGenerator']

    for alph in list(scores.keys()):
        avg_coeff_determination[alph] = np.median(
            coeff_of_determination[alph]
        )
        if np.isnan(avg_coeff_determination[alph]):
            # don't allow non-numerical R^2
            # happens when sum of squares =0 in calculation,
            # because true_exp_val=1 for all times, so no variance
            avg_coeff_determination[alph] = 0

    unique_growth_rules = list(
        set(list(growth_rules.values()))
    )

    growth_classes = {}
    for g in list(growth_rules.keys()):
        try:
            growth_classes[g] = unique_growth_classes[growth_rules[g]]
        except BaseException:
            growth_classes[g] = None

    latex_f_scores = {}
    latex_coeff_det = {}
    wins = {}
    for mod in list(scores.keys()):
        latex_name = unique_growth_classes[growth_rules[mod]].latex_name(mod)
        latex_f_scores[latex_name] = f_scores[mod]
        latex_coeff_det[latex_name] = avg_coeff_determination[mod]
        precisions[latex_name] = precisions[mod]
        precisions.pop(mod)
        sensitivities[latex_name] = sensitivities[mod]
        sensitivities.pop(mod)
        wins[latex_name] = scores[mod]
        model_results[latex_name] = {
            'precision': precisions[latex_name],
            'sensitivity': sensitivities[latex_name],
            'f_score': latex_f_scores[latex_name],
            'median_r_squared': latex_coeff_det[latex_name],
            'r_squared_individual_instances': coeff_of_determination[mod],
            'volumes': volumes[mod]
        }

    results = {
        'scores': scores,
        'growth_rules': growth_rules,
        'growth_classes': growth_classes,
        'unique_growth_classes': unique_growth_classes,
        'avg_coeff_determination': avg_coeff_determination,
        'f_scores': latex_f_scores,
        'latex_coeff_det': latex_coeff_det,
        'precisions': precisions,
        'sensitivities': sensitivities,
        'wins': wins
    }

    if collective_analysis_pickle_file is not None:
        if os.path.isfile(collective_analysis_pickle_file) is False:
            print(
                "[get_model_scores] Saving collective analysis. \nfile:",
                collective_analysis_pickle_file)
            pickle.dump(
                model_results,
                open(collective_analysis_pickle_file, 'wb')
            )
        else:
            # load current analysis dict, add to it and rewrite it.
            combined_analysis = pickle.load(
                open(collective_analysis_pickle_file, 'rb')
            )

            for model in list(model_results.keys()):
                for res in list(model_results[model].keys()):
                    try:
                        combined_analysis[model][res] = model_results[model][res]
                    except BaseException:
                        combined_analysis[model] = {
                            res: model_results[model][res]
                        }
            pickle.dump(
                combined_analysis,
                open(collective_analysis_pickle_file, 'wb')
            )

    return results


def get_entropy(
    models_points,
    growth_generator=None,
    inf_gain=False
):
    """
    Deprecated -- using old functionality e.g. get_all_model_names 
    should come from growth class; 
    but retained in case plotting logic wanted later
    """
    # TODO this calculation of entropy may not be correct
    # What is initial_entropy meant to be?
    num_qmd_instances = sum(list(models_points.values()))
    num_possible_qmd_instances = len(
        get_all_model_names(
            growth_generator=growth_generator,
            return_branch_dict='latex_terms'
        )
    )
    # TODO don't always want ising terms only

    model_fractions = {}
    for k in list(models_points.keys()):
        model_fractions[k] = models_points[k] / num_qmd_instances

    initial_entropy = -1 * np.log2(1 / num_possible_qmd_instances)
    entropy = 0
    for i in list(models_points.keys()):
        success_prob = model_fractions[i]
        partial_entropy = success_prob * np.log2(success_prob)
        if np.isnan(partial_entropy):
            partial_entropy = 0
        entropy -= partial_entropy

    if inf_gain:
        # information gain is entropy loss
        information_gain = initial_entropy - entropy
        return information_gain
    else:
        return entropy


def plot_scores(
    scores,
    growth_classes,
    unique_growth_classes,
    growth_rules,
    coefficients_of_determination=None,
    coefficient_determination_latex_name=None,
    f_scores=None,
    plot_r_squared=True,
    plot_f_scores=False,
    entropy=None,
    inf_gain=None,
    true_model=None,
    growth_generator=None,
    batch_nearest_num_params_as_winners=True,
    collective_analysis_pickle_file=None,
    save_file='model_scores.png'
):
    plt.clf()
    models = list(scores.keys())

    # print("[AnalyseMultiple - plot_scores] growth classes:",growth_classes )
    # print("[AnalyseMultiple - plot_scores] unique_growth_classes:",unique_growth_classes )
    latex_true_op = unique_growth_classes[growth_generator].latex_name(
        name=true_model
    )

    latex_model_names = [
        growth_classes[model].latex_name(model)
        for model in models
    ]
    print(
        "[multiQMD plots]coefficients_of_determination:",
        coefficients_of_determination)
    print("[multiQMD plots]f scores:", f_scores)

    # coefficient_determination_latex_name = {}
    # f_score_latex_name = {}
    # for mod in list(coefficients_of_determination.keys()):
    #     coefficient_determination_latex_name[
    #         growth_classes[mod].latex_name(mod)
    #     ] = coefficients_of_determination[mod]

    #     f_score_latex_name[
    #         growth_classes[mod].latex_name(mod)
    #     ] = f_scores[mod]

    coeff_of_determination = [
        coefficient_determination_latex_name[latex_mod]
        for latex_mod in latex_model_names
    ]

    f_scores_list = [
        f_scores[latex_mod]
        for latex_mod in latex_model_names
    ]

    latex_scores_dict = {}
    for mod in models:
        latex_mod = growth_classes[mod].latex_name(mod)
        latex_scores_dict[latex_mod] = scores[mod]

    batch_correct_models = []
    if batch_nearest_num_params_as_winners == True:
        num_true_params = len(
            database_framework.get_constituent_names_from_name(
                true_model
            )
        )
        for mod in models:
            num_params = len(
                database_framework.get_constituent_names_from_name(mod)
            )

            if (
                num_true_params - num_params == 1
            ):
                # must be exactly one parameter smaller
                batch_correct_models.append(
                    mod
                )

    mod_scores = scores
    scores = list(scores.values())
    num_runs = sum(scores)

    # fig, ax = plt.subplots()
    width = 0.75  # the width of the bars
    ind = np.arange(len(scores))  # the x locations for the groups
    colours = ['blue' for i in ind]
    batch_success_rate = correct_success_rate = 0
    for mod in batch_correct_models:
        mod_latex = growth_classes[mod].latex_name(mod)
        mod_idx = latex_model_names.index(mod_latex)
        colours[mod_idx] = 'orange'
        batch_success_rate += mod_scores[mod]
    if true_model in models:
        batch_success_rate += mod_scores[true_model]
        correct_success_rate = mod_scores[true_model]

    batch_success_rate /= num_runs
    correct_success_rate /= num_runs
    batch_success_rate *= 100
    correct_success_rate *= 100  # percent

    results_collection = {
        'type': growth_generator,
        'true_model': latex_true_op,
        'scores': latex_scores_dict
    }
    print("[Analyse] results_collection", results_collection)

    # if save_results_collection is not None:
    #     print("[Analyse] save results collection:", save_results_collection)
    #     pickle.dump(
    #         results_collection,
    #         open(
    #             save_results_collection,
    #             'wb'
    #         )
    #     )
    if collective_analysis_pickle_file is not None:
        # no longer used/accessed by this function
        if os.path.isfile(collective_analysis_pickle_file) is False:
            combined_analysis = {
                'scores': results_collection
            }
            pickle.dump(
                combined_analysis,
                open(collective_analysis_pickle_file, 'wb')
            )
        else:
            # load current analysis dict, add to it and rewrite it.
            combined_analysis = pickle.load(
                open(collective_analysis_pickle_file, 'rb')
            )
            combined_analysis['scores'] = results_collection
            pickle.dump(
                combined_analysis,
                open(collective_analysis_pickle_file, 'wb')
            )

    try:
        true_idx = latex_model_names.index(
            latex_true_op
        )
        colours[true_idx] = 'green'

    except BaseException:
        pass

    fig, ax1 = plt.subplots(
        figsize=(
            max(max(scores), 5),
            max((len(scores) / 4), 3)
        )
    )

    # ax.barh(ind, scores, width, color="blue")
    ax1.barh(ind, scores, width, color=colours)
    ax1.set_yticks(ind + width / 2)
    ax1.set_yticklabels(
        latex_model_names,
        minor=False
    )
    ax1.set_xlabel('Number wins')
    xticks_pos = list(range(max(scores) + 1))
    ax1.set_xticks(
        xticks_pos,
        minor=False
    )
    custom_lines = [
        Line2D([0], [0], color='green', lw=4),
        Line2D([0], [0], color='orange', lw=4),
        Line2D([0], [0], color='blue', lw=4),
        # Line2D([0], [0], color='black', lw=4, ls='--'),
    ]
    custom_handles = [
        'True ({}%)'.format(int(correct_success_rate)),
        'True/Close ({}%)'.format(int(batch_success_rate)),
        'Other',
        # '$R^2$'
    ]

    if plot_r_squared == True:
        ax2 = ax1.twiny()
        ax2.barh(
            ind,
            coeff_of_determination,
            width / 2,
            color=colours,
            label='$R^2$',
            linestyle='--',
            fill=False,
        )
        # ax2.invert_xaxis()
        ax2.set_xlabel('$R^2$')
        ax2.xaxis.tick_top()

        r_sq_x_ticks = [
            min(coeff_of_determination),
            0,
            1
        ]
        ax2.set_xticks(r_sq_x_ticks)
        ax2.legend(
            bbox_to_anchor=(1.0, 0.9),
        )
    elif plot_f_scores == True:
        ax2 = ax1.twiny()
        ax2.barh(
            ind,
            f_scores_list,
            width / 2,
            color=colours,
            label='F-score',
            linestyle='--',
            fill=False,
        )
        # ax2.invert_xaxis()
        ax2.set_xlabel('F-score')
        ax2.xaxis.tick_top()

        f_score_x_ticks = [
            # min(coeff_of_determination),
            0,
            1
        ]
        ax2.set_xticks(f_score_x_ticks)
        ax2.legend(
            bbox_to_anchor=(1.0, 0.9),
        )

    plot_title = str(
        'Number of QMD instances won by models with $R^2$.'
    )

    if entropy is not None:
        plot_title += str(
            r'\n$\mathcal{S}$='
            + str(round(entropy, 2))
        )
    if inf_gain is not None:
        plot_title += str(
            r'\t $\mathcal{IG}$='
            + str(round(inf_gain, 2))
        )
    ax1.legend(
        custom_lines,
        custom_handles,
        bbox_to_anchor=(1.0, 0.4),
    )

    # plt.legend(
    #     custom_lines,
    #     custom_handles
    # )
    # plt.title(plot_title)
    plt.ylabel('Model')
    # plt.xlabel('Number of wins')
    #plt.bar(scores, latex_model_names)

    plt.savefig(save_file, bbox_inches='tight')


def stat_metrics_histograms(
    champ_info, 
    save_to_file=None
):

    include_plots = [
        {'name' : 'f_scores', 'colour' : 'red'}, 
        {'name' : 'precisions',  'colour': 'blue'}, 
        {'name' : 'sensitivities', 'colour' : 'green'}, 
    ]

    fig = plt.figure(
        figsize=(15, 5),
        tight_layout=True
    )
    gs = GridSpec(
        nrows=1,
        ncols=len(include_plots),
        # figure=fig # not available on matplotlib 2.1.1 (on BC)
    )
    plot_col = 0
    hist_bins = np.arange(0,1, 0.1)
    for plotting_data in include_plots: 
        ax = fig.add_subplot(gs[0, plot_col])
        data = champ_info[plotting_data['name']]
        print("data for {}: {}".format(plotting_data['name'], data))
        ax.hist(
            list(data.values()), 
            color = plotting_data['colour'],
            bins = hist_bins
        )
        ax.set_xlim(0,1)    
        ax.set_title(
            "Champions' {}".format(plotting_data['name'])
        )
        ax.set_xlabel(plotting_data['name'])
        ax.set_ylabel('# Champions')
        plot_col += 1
    if save_to_file is not None: 
        plt.savefig(save_to_file)

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
    try:
        # qmd_res = pandas.DataFrame.from_csv(
        qmd_res = pandas.read_csv(
            results_csv,
            index_col='LatexName'
        )
    except ValueError:
        print(
            "Latex Name not in results CSV keys.",
            "There aren't enough data for a tree of multiple QMD."
            "This may be because this run was for QHL rather than QMD."
        )
        raise

    mods = list(qmd_res.index)
    winning_count = {}
    for mod in mods:
        winning_count[mod] = mods.count(mod)

    cumulativeQMDTreePlot(
        cumulative_csv=all_bayes_csv,
        wins_per_mod=winning_count,
        latex_mapping_file=latex_mapping_file,
        growth_generator=growth_generator,
        only_adjacent_branches=True,
        avg=avg_type,
        entropy=entropy,
        inf_gain=inf_gain,
        save_to_file=save_to_file
    )


def count_model_occurences(
    latex_map,
    true_model_latex,
    save_counts_dict=None,
    save_to_file=None
):
    f = open(latex_map, 'r')
    l = str(f.read())
    terms = l.split("',")

    # for t in ["(", ")", "'", " "]:
    for t in ["'", " "]:
        terms = [a.replace(t, '') for a in terms]

    sep_terms = []
    for t in terms:
        sep_terms.extend(t.split("\n"))

    unique_models = list(set([s for s in sep_terms if "$" in s]))
    counts = {}
    for ln in unique_models:
        counts[ln] = sep_terms.count(ln)
    unique_models = sorted(unique_models)
    model_counts = [counts[m] for m in unique_models]
    unique_models = [
        a.replace("\\\\", "\\")
        for a in unique_models
    ]  # in case some models have too many slashes.
    max_count = max(model_counts)
    integer_ticks = list(range(max_count + 1))
    colours = ['blue' for m in unique_models]
    unique_models = [u[:-1] for u in unique_models if u[-1] == ')']
    true_model_latex = true_model_latex.replace(' ', '')
    if true_model_latex in unique_models:
        true_idx = unique_models.index(true_model_latex)
        colours[true_idx] = 'green'

    print(
        "[multiQMD - count model occurences]",
        "Colours:", colours,
        "\ntrue op:", true_model_latex,
        "\nunique models:", unique_models,
        "test:", (str(true_model_latex) in unique_models)
    )

    fig, ax = plt.subplots(
        figsize=(
            max(max_count * 2, 5),
            len(unique_models) / 2)
    )
    ax.plot(kind='barh')
    ax.barh(
        unique_models,
        model_counts,
        color=colours
    )
    ax.set_xticks(integer_ticks)
    ax.set_title('# times each model generated')
    ax.set_xlabel('# occurences')
    ax.tick_params(
        top=True,
        direction='in'
    )
    if save_counts_dict is not None:
        import pickle
        pickle.dump(
            counts,
            open(
                save_counts_dict,
                'wb'
            )
        )

    try:
        if save_to_file is not None:
            plt.savefig(
                save_to_file,
                bbox_inches='tight'
            )
    except BaseException:
        print(
            "[AnalyseMultiple - count model occurences] couldn't save plot to file",
            save_to_file

        )
        raise


def plot_statistics(
    to_plot,  # list of dictionaries of statistics to plot
    models,  # list of models to plot results for,
    true_model,
    colourmap=plt.cm.tab20c,
    save_to_file=None
):

    num_models = len(models)
    num_cases = len(to_plot)
    widths = {}
    b = 0
    w = 0
    width = 1 / (num_cases + 1)
    for l in to_plot:
        l['width'] = w
        w -= width

    indices = np.arange(num_models)
    cm_subsection = np.linspace(0, 0.8, len(to_plot))
    colours = [colourmap(x) for x in cm_subsection]
    custom_lines = []
    custom_handles = []
    max_top_range = 0

    plt.clf()
    fig, top_ax = plt.subplots(
        figsize=(15, 2 * len(models))
    )

    bottom_ax = top_ax.twiny()

    top_ax.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False
    )
    bottom_ax.tick_params(
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True
    )

    top_labels = []
    bottom_labels = []
    alpha = 1
    fill = True
    for dataset in to_plot:
        if dataset['range'] == 'cap_1':
            ax = bottom_ax
            bottom_labels.append(dataset['title'])
            ls = '-'
            fill = True
            this_width = width
            alpha = 1
            lw = 2
        else:
            ax = top_ax
            top_labels.append(dataset['title'])
            if dataset['title'] == '# Wins':
                top_colour = colours[to_plot.index(dataset)]
                this_width = 1 * width
                alpha = 1.0
                ls = '--'
                lw = 3
                fill = True
#                 this_width = num_cases* width ## to make this bar go underneath all stats for this model
#                 dataset['width'] = 0.5
        res = dataset['res']
        colour = colours[to_plot.index(dataset)]
        ax.barh(
            indices + dataset['width'],
            res,
            this_width,
            color=colour,
            alpha=alpha,
            ls=ls,
            fill=fill,
            linewidth=lw
        )

        custom_lines.append(
            Line2D([0], [0], color=colour, lw=4),
        )
        custom_handles.append(dataset['title'])

    label_size = 25

    bottom_ax.set_xlabel(
        '; '.join(bottom_labels),
        fontsize=label_size,
    )
    bottom_ax.xaxis.set_label_position('bottom')
    bottom_ax.set_xticks([0, 0.5, 1])
    bottom_ax.set_xticklabels(
        [0, 0.5, 1],
        fontsize=label_size
    )

    top_ax.set_xlabel(
        '; '.join(top_labels),
        fontsize=label_size,
        color=top_colour,
        fontweight='bold'
    )
    top_ax.xaxis.set_label_position('top')
    top_ax.xaxis
    xticks = list(set([int(i) for i in top_ax.get_xticks()]))
    top_ax.set_xticks(
        xticks,
    )
    top_ax.set_xticklabels(
        xticks,
        fontsize=label_size
    )

    min_xlim = min(min(bottom_ax.get_xlim()), 0)
    max_xlim = max(max(top_ax.get_xlim()), 1)
    xlim = (min_xlim, max_xlim)
    top_ax.set_xlim(xlim)

    top_ax.set_yticklabels(models, fontsize=label_size)
    top_ax.set_yticks(indices + 0.1)
    model_label_colours = ['black' for m in models]

    try:
        true_idx = models.index(str(true_model))
        top_ax.get_yticklabels()[true_idx].set_color('green')
    except BaseException:
        pass

    plt.legend(
        custom_lines,
        custom_handles,
        bbox_to_anchor=(1.0, 0.4),
        fontsize=20
    )

    if save_to_file is not None:
        plt.savefig(
            save_to_file,
            bbox_inches='tight'
        )


def summarise_qmla_text_file(
    results_csv_path, 
    path_to_summary_file
):
    all_results = pandas.read_csv(results_csv_path)

    to_write = "\
        {num_true_found} instance(s) total \n\
        True model won {true_mod_found} instance(s); considered in {true_mod_considered} instance(s). \n\
        Average time taken: {avg_time} seconds \n\
        True growth rules: {growth_rules} \n\
        Min/median/max number of models per instance: {min_num_mods}/{median_num_mods}/{max_num_mods}. \n\
        ".format(
            num_true_found = len(all_results), 
            true_mod_considered = all_results['TrueModelConsidered'].sum(), 
            true_mod_found = all_results['TrueModelFound'].sum(),
            avg_time = np.round(all_results['Time'].median(), 2),
            growth_rules = list(all_results.GrowthGenerator.unique()),
            min_num_mods = all_results['NumModels'].min(),
            median_num_mods = all_results['NumModels'].median(),
            max_num_mods = all_results['NumModels'].max()
        )

    with open(path_to_summary_file, 'w') as summary_file:
        print(
            to_write, 
            file=summary_file, 
            flush=True
        )

def avg_f_score_multi_qmla(
    results_csv_path,
    save_to_file=None
):
    plt.clf()
    all_results = pandas.read_csv(results_csv_path)
    gen_f_scores = all_results.GenerationalFscore

    all_f_scores = None
    for g in gen_f_scores.index:
        data = eval(gen_f_scores[g])
        indices = list(data.keys())
        data_array = np.array(
            [data[i] for i in indices]
        )
        p = pandas.DataFrame(
            data_array, 
            columns=['Fscore'],
            index=indices
        )
        p['ID'] = g
        p['Gen'] = indices

        if all_f_scores is None:
            all_f_scores = p
        else:
            all_f_scores = all_f_scores.append(p, ignore_index=True)


    try:
        avg_f_scores = [
            np.median(
                flatten(list(all_f_scores[all_f_scores['Gen'] == g].Fscore))
            )        
            for g in indices
        ]
        lower_quartile = [
            np.percentile(
                flatten(list(all_f_scores[all_f_scores['Gen'] == g].Fscore)), 
                25
            )        
            for g in indices
        ]
        upper_quartile = [
            np.percentile(
                flatten(list(all_f_scores[all_f_scores['Gen'] == g].Fscore)), 
                75
            )        
            for g in indices    
        ]
    except: 
        print("Not enough data for multiple plot points.")
        f = list(all_f_scores[all_f_scores['Gen'] == 1].Fscore)

        print("Indices:", indices)
        avg_f_scores = [np.median(f)]
        lower_quartile = [np.percentile(f, 25)]
        upper_quartile = [np.percentile(f, 75)]

    plt.plot(
        indices, 
        avg_f_scores,
        marker='o',
        label='Median F score'
    )

    plt.fill_between(
        indices, 
        lower_quartile, 
        upper_quartile,
        label='Inter-quartile range',
        alpha=0.2
    )
    plt.title("Median F-score V QMLA generation")
    plt.ylabel('F-score')
    plt.xlabel('Generation')
    plt.ylim(-0.1, 1.1)
    plt.yticks([0,0.25, 0.5, 0.75, 1])
    plt.xticks(indices)
    plt.legend()
    
    if save_to_file is not None: 
        plt.savefig(
            save_to_file
        )
    print("Plotted average f scores by generation")
