import numpy as np
import argparse
from matplotlib.lines import Line2D
import sys
import os
import pickle
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from qmla.analysis.analysis_and_plot_functions import fill_between_sigmas, cumulativeQMDTreePlot
import qmla.get_growth_rule as get_growth_rule
import qmla.model_naming as model_naming
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

    # qmd_cumulative_results = pd.DataFrame.from_csv(results_csv,
    qmd_cumulative_results = pd.read_csv(results_csv,
                                                       index_col='ConfigLatex'
                                                       )
    piv = pd.pivot_table(
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

    # results = pd.DataFrame.from_csv(
    results = pd.read_csv(
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
    latex_model_wins = {}
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
        latex_model_wins[latex_name] = scores[mod]
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
        'latex_model_wins' : latex_model_wins, 
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
                np.abs(num_true_params - num_params) == 1
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
    hist_bins = np.arange(0, 1.01,0.05)
    model_wins = champ_info['latex_model_wins']
    for plotting_data in include_plots: 
        ax = fig.add_subplot(gs[0, plot_col])
        data = champ_info[plotting_data['name']]
        weights = {}
        for mod in model_wins: 
            d = data[mod]
            wins = model_wins[mod]
            try:
                weights[d] += wins
            except: 
                weights[d] = wins

        values_to_plot = sorted(weights.keys())
        wts = [weights[v] for v in values_to_plot]
        ax.hist(
            values_to_plot, 
            weights = wts, 
            color = plotting_data['colour'],
            bins = hist_bins,
            align='mid'
        )
        ax.set_xlim(0,1.0)    
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
        # qmd_res = pd.DataFrame.from_csv(
        qmd_res = pd.read_csv(
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
    all_results = pd.read_csv(results_csv_path)

    to_write = "\
        {num_instances} instance(s) total. \n\
        True model won {true_mod_found} instance(s); considered in {true_mod_considered} instance(s). \n\
        {n_exp} experiments; {n_prt} particles. \n\
        Average time taken: {avg_time} seconds. \n\
        True growth rules: {growth_rules}. \n\
        Min/median/max number of models per instance: {min_num_mods}/{median_num_mods}/{max_num_mods}. \n\
        ".format(
            num_instances = len(all_results), 
            n_exp = int(all_results['NumExperiments'].mean()),
            n_prt = int(all_results['NumParticles'].mean()),
            true_mod_considered = all_results['TrueModelConsidered'].sum(), 
            true_mod_found = all_results['TrueModelFound'].sum(),
            avg_time = np.round(all_results['Time'].median(), 2),
            growth_rules = list(all_results.GrowthGenerator.unique()),
            min_num_mods = int(all_results['NumModels'].min()),
            median_num_mods = int(all_results['NumModels'].median()),
            max_num_mods = int(all_results['NumModels'].max())
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
    all_results = pd.read_csv(results_csv_path)
    gen_f_scores = all_results.GenerationalFscore
    gen_log_likelihoods = all_results.GenerationalLogLikelihoods

    all_f_scores = None
    all_log_likelihoods = None
    for g in gen_f_scores.index:
        data = eval(gen_f_scores[g])
        log_lk = eval(gen_log_likelihoods[g])

        indices = list(data.keys())
        data_array = np.array(
            [data[i] for i in indices]
        )
        p = pd.DataFrame(
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

def generational_analysis(combined_results, save_directory=None):

    generational_scores = None

    for k in combined_results.index:
        single_instance_gen_ll = eval(combined_results['GenerationalLogLikelihoods'][k])
        single_instance_gen_f_score= eval(combined_results['GenerationalFscore'][k])

        for gen in list(single_instance_gen_ll.keys()):
            this_gen_ll = single_instance_gen_ll[gen]
            this_gen_log_abs_ll = [np.log(np.abs(ll)) for ll in this_gen_ll]
            this_gen_f_score = single_instance_gen_f_score[gen]
            this_gen_data = list(zip(this_gen_ll, this_gen_log_abs_ll, this_gen_f_score))

            df = pd.DataFrame(
                data = this_gen_data,
                columns = ['log_likelihood', 'log_abs_ll', 'f_score']
            )

            df['gen'] = gen
            df['instance'] = k
            if generational_scores is None:
                generational_scores = df
            else: 
                generational_scores = generational_scores.append(
                    df, 
                    ignore_index=True
                )

    num_instances = len(generational_scores.instance.unique())
    fig = plt.figure(
        figsize=(15, 8),
        tight_layout=True
    )
    gs = GridSpec(
        2,
        1,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    
    sns.boxplot(
        data = generational_scores, 
        x = 'gen',
        y = 'f_score',
        showfliers=False, 
        ax = ax1, 
    )
    sns.swarmplot(
        data = generational_scores, 
        x = 'gen',
        y = 'f_score',
        # showfliers=False, 
        color='grey',
        ax = ax1, 
    )

    ax1.set_ylabel('F score')
    ax1.set_xlabel('Generation')
    ax1.set_title('F score V Generation')
    ax1.set_ylim(0,1)
    ax1.axhline(0.5, ls='--',color='black')

    sns.pointplot(
        data = generational_scores, 
        x = 'gen',
        y = 'f_score',
#         showfliers=False, 
        hue='instance',
        ci=None, 
        ax = ax2, 
    )
    ax2.set_title("F score individual instances")
    ax2.axhline(0.5, ls='--', color='black')
    ax2.set_ylim(0,1)
    ax2.legend(
        title='Instance',
        ncol=min(8, num_instances)
    )

    if save_directory is not None:
        plt.savefig(
            os.path.join(
                save_directory, 
                "generational_measures_f_scores.png"
            )
        )
        
    plt.clf()
    gs = GridSpec(
        2,
        1,
    )
    ax3 = fig.add_subplot(gs[0, 0])
    ax4 = fig.add_subplot(gs[1, 0])

    sns.boxplot(
        data = generational_scores, 
        x = 'gen',
        y = 'log_likelihood',
        showfliers=False, 
        ax = ax3, 
    )

    ax3.set_title('Log likelihood V generation')
    ax3.set_ylabel('Log likelihood')
    ax3.set_xlabel('Generation')

    sns.pointplot(
        data = generational_scores, 
        x = 'gen',
        y = 'log_likelihood',
        # showfliers=False, 
        ci=None, 
        hue='instance',
        ax = ax4, 
    )

    ax4.set_title('Log likelihood individual instances')
    ax4.set_ylabel('Log likelihood')
    ax4.set_xlabel('Generation')
    ax4.legend(
        title='Instance',
        ncol=min(8, num_instances)
    )

    if save_directory is not None:
        plt.savefig(
            os.path.join(
                save_directory, 
                "generational_measures_log_likelihoods.png"
            )
        )