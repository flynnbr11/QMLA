import numpy as np
import argparse
import sys
import os
import pickle
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

import qmla.database_framework

__all__ = [
    'update_shared_bayes_factor_csv', 
    'plot_scores',
    'stat_metrics_histograms',
    'parameter_sweep_analysis'
]

def update_shared_bayes_factor_csv(qmd, all_bayes_csv):
    import os
    import csv
    data = get_bayes_latex_dict(qmd)
    names = list(data.keys())
    fields = ['ModelName']
    fields += names

    all_models = []
    if os.path.isfile(all_bayes_csv) is False:
        # all_models += ['ModelName']
        # all_models += names
        # print("file exists:", os.path.isfile(all_bayes_csv))
        # print("creating CSV")
        with open(all_bayes_csv, 'a+') as bayes_csv:
            writer = csv.DictWriter(
                bayes_csv,
                fieldnames=fields
            )
            writer.writeheader()
    else:
        # print("file exists:", os.path.isfile(all_bayes_csv))
        current_csv = csv.DictReader(open(all_bayes_csv))
        current_fieldnames = current_csv.fieldnames
        new_models = list(
            set(fields) - set(current_fieldnames)
        )

        if len(new_models) > 0:

            import pandas
            csv_input = pandas.read_csv(
                all_bayes_csv,
                index_col='ModelName'
            )
            a = list(csv_input.keys())
            # print("pandas says existing models are:\n", a)
            empty_list = [np.NaN] * len(list(csv_input[a[0]].values))

            for new_col in new_models:
                csv_input[new_col] = empty_list
            # print("writing new pandas CSV: ", csv_input)
            csv_input.to_csv(all_bayes_csv)

    with open(all_bayes_csv) as bayes_csv:
        reader = csv.DictReader(
            bayes_csv,
        )
        fields = reader.fieldnames

    with open(all_bayes_csv, 'a') as bayes_csv:
        writer = csv.DictWriter(
            bayes_csv,
            fieldnames=fields,
        )
        for f in names:
            single_model_dict = data[f]
            single_model_dict['ModelName'] = f
            writer.writerow(single_model_dict)


def get_bayes_latex_dict(qmd):
    latex_dict = {}
    # print("get bayes latex dict")

    latex_write_file = open(
        # str(qmd.results_directory + 'LatexMapping.txt'),
        qmd.latex_name_map_file_path,
        'a+'
    )
    for i in list(qmd.all_bayes_factors.keys()):
        mod = qmd.model_name_id_map[i]
        latex_name = qmd.get_model_storage_instance_by_id(i).model_name_latex
        mapping = (mod, latex_name)
        print(mapping, file=latex_write_file)

    for i in list(qmd.all_bayes_factors.keys()):
        mod_a = qmd.get_model_storage_instance_by_id(i).model_name_latex
        latex_dict[mod_a] = {}
        for j in list(qmd.all_bayes_factors[i].keys()):
            mod_b = qmd.get_model_storage_instance_by_id(j).model_name_latex
            latex_dict[mod_a][mod_b] = qmd.all_bayes_factors[i][j][-1]
    return latex_dict


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
    latex_true_op = unique_growth_classes[growth_generator].latex_name(
        name=true_model
    )

    latex_model_names = [
        growth_classes[model].latex_name(model)
        for model in models
    ]
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
            qmla.database_framework.get_constituent_names_from_name(
                true_model
            )
        )
        for mod in models:
            num_params = len(
                qmla.database_framework.get_constituent_names_from_name(mod)
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
