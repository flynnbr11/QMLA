import numpy as np
import argparse
import sys
import os
import pickle
import csv
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import scipy
import lfig 

import qmla.construct_models

__all__ = [
    'bayes_factor_f_score_heatmap',
    'cross_instance_bayes_factor_heatmap',
    'update_shared_bayes_factor_csv', 
    'get_model_scores', 
    'plot_statistics', 
    'summarise_qmla_text_file',
    'plot_scores',
    'stat_metrics_histograms',
    'parameter_sweep_analysis',
    'plot_evaluation_log_likelihoods',
    'count_term_occurences',
    'inspect_times_on_nodes'
]

def bayes_factor_f_score_heatmap(
    bayes_factors_df,
    save_to_file=None,
):
    lf = lfig.LatexFigure()
    ax1 = lf.new_axis()

    bayes_factor_by_f_score = pd.pivot_table(
        bayes_factors_df, 
        values='log10_bayes_factor', 
        index=['f_score_a'], 
        columns=['f_score_b'],
        aggfunc=np.median
    )
    print("bayes_factor_by_id: \n")

    mask = np.tri(bayes_factor_by_f_score.shape[0], k=0).T
    sns.heatmap(
        bayes_factor_by_f_score,
        cmap=matplotlib.cm.PRGn, # TODO get from ES?
        mask=mask,
        annot=True, 
        ax = ax1,
        cbar_kws={
            "orientation": "vertical",
            "label" : r"$\log_{10}\left(B_{a,b}\right)$"
        }
    )
    ax1.set_ylabel('$F(a)$')
    ax1.set_xlabel('$F(b)$')
    ax1.set_title('$F(A) > F(B)$')

    lf.fig.suptitle(r"$\log_{10}$ Bayes factor by F score", fontsize=25, y=1.15)

    if save_to_file is not None:
        lf.save(save_to_file)


def cross_instance_bayes_factor_heatmap(
    combined_datasets_directory, 
    save_directory
):
    try:
        fitness_correlations = pd.read_csv(
            os.path.join(
                combined_datasets_directory, 
                'fitness_correlations.csv'
            )
        )
    except:
        print("ANALYSIS FAILURE: could not load fitness correlation CSV.")
        pass # allow to fail # todo just return exception

    bayes_factors = pd.read_csv(
        os.path.join(combined_datasets_directory, 'bayes_factors.csv')
    )
    f = bayes_factor_f_score_heatmap(
        bayes_factors, 
        save_to_file = os.path.join(save_directory, 'bayes_factors_by_f_scores')
    )
    # f.savefig(
    # )


def update_shared_bayes_factor_csv(qmd, all_bayes_csv):
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

def get_model_scores(
    directory_name,
    unique_exploration_classes,
    collective_analysis_pickle_file=None,
):

    os.chdir(directory_name)

    scores = {}
    exploration_strategies = {}

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

        if alph not in list(exploration_strategies.keys()):
            exploration_strategies[alph] = result['ExplorationRule']

    for alph in list(scores.keys()):
        avg_coeff_determination[alph] = np.median(
            coeff_of_determination[alph]
        )
        if np.isnan(avg_coeff_determination[alph]):
            # don't allow non-numerical R^2
            # happens when sum of squares =0 in calculation,
            # because true_exp_val=1 for all times, so no variance
            avg_coeff_determination[alph] = 0

    unique_exploration_strategies = list(
        set(list(exploration_strategies.values()))
    )

    exploration_classes = {}
    for g in list(exploration_strategies.keys()):
        try:
            exploration_classes[g] = unique_exploration_classes[exploration_strategies[g]]
        except BaseException:
            exploration_classes[g] = None

    latex_f_scores = {}
    latex_coeff_det = {}
    wins = {}
    for mod in list(scores.keys()):
        latex_name = unique_exploration_classes[exploration_strategies[mod]].latex_name(mod)
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
        'exploration_strategies': exploration_strategies,
        'exploration_classes': exploration_classes,
        'unique_exploration_classes': unique_exploration_classes,
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
            if dataset['title'] == 'Number Wins':
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
        True exploration strategies: {exploration_strategies}. \n\
        Min/median/max number of models per instance: {min_num_mods}/{median_num_mods}/{max_num_mods}. \n\
        ".format(
            num_instances = len(all_results), 
            n_exp = int(all_results['NumExperiments'].mean()),
            n_prt = int(all_results['NumParticles'].mean()),
            true_mod_considered = all_results['TrueModelConsidered'].sum(), 
            true_mod_found = all_results['TrueModelFound'].sum(),
            avg_time = np.round(all_results['Time'].median(), 2),
            exploration_strategies = list(all_results.ExplorationRule.unique()),
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


def plot_scores(
    scores,
    exploration_classes,
    unique_exploration_classes,
    exploration_strategies,
    coefficients_of_determination=None,
    coefficient_determination_latex_name=None,
    f_scores=None,
    plot_r_squared=True,
    plot_f_scores=False,
    entropy=None,
    inf_gain=None,
    true_model=None,
    exploration_rule=None,
    batch_nearest_num_params_as_winners=True,
    collective_analysis_pickle_file=None,
    save_file='model_scores.png'
):
    plt.clf()
    models = list(scores.keys())
    latex_true_op = unique_exploration_classes[exploration_rule].latex_name(
        name=true_model
    )

    latex_model_names = [
        exploration_classes[model].latex_name(model)
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
        latex_mod = exploration_classes[mod].latex_name(mod)
        latex_scores_dict[latex_mod] = scores[mod]

    batch_correct_models = []
    if batch_nearest_num_params_as_winners == True:
        num_true_params = len(
            qmla.construct_models.get_constituent_names_from_name(
                true_model
            )
        )
        for mod in models:
            num_params = len(
                qmla.construct_models.get_constituent_names_from_name(mod)
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
        mod_latex = exploration_classes[mod].latex_name(mod)
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
        'type': exploration_rule,
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
        r'True ({}$\%$)'.format(int(correct_success_rate)),
        r'True/Close ({}$\%$)'.format(int(batch_success_rate)),
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
        ax.set_ylabel('Number Champions')
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


def round_nearest(x,a):
    return round(round(x/a)*a ,2)

def plot_evaluation_log_likelihoods(
    combined_results, 
    include_log_likelihood = True,
    include_median_likelihood = True,
    include_champion_percentiles = True,
    save_data_as_csv=False, 
    save_directory=None,
):
    evaluation_cols = [
        'instance', 'model_id', 
        'log_likelihood', 'median_likelihood',
        'likelihood_percentile',
        'f_score', 'true', 'champ', 'Classification'
    ]
    evaluation_plot_df = pd.DataFrame(
        columns = evaluation_cols
    )
    for i in list(combined_results.index):
        res = combined_results.iloc[i]

        log_lls = eval(res.ModelEvaluationLogLikelihoods)
        raw_lls = list(log_lls.values())
        median_lls = eval(res.ModelEvaluationMedianLikelihoods)
        f_scores = eval(res.AllModelFScores)

        instance = res.QID
        instance_true_id = res.Truemodel_id
        instance_champion_id = res.ChampID

        model_ids = list(log_lls.keys())
        for mod in model_ids: 
            if (
                mod == instance_true_id
                and
                mod == instance_champion_id
            ):
                model_classification = 'True + Champion'
            elif mod == instance_true_id: 
                model_classification = 'True'
            elif mod == instance_champion_id: 
                model_classification = 'Champion'
            else:
                model_classification = 'Standard'

            ll_percentile = scipy.stats.percentileofscore(
                a = raw_lls, 
                score = log_lls[mod],
                kind='weak'
            )
            ll_percentile = round_nearest(ll_percentile, 5)
            
            this_mod_df = pd.DataFrame(
                [[
                    # i, # for some reason instance causes a shift between the two plot types?
                    instance, 
                    mod, 
                    log_lls[mod],
                    median_lls[mod],
                    ll_percentile, 
                    f_scores[mod],
                    mod==instance_true_id,
                    mod==instance_champion_id,
                    model_classification, 
                ]],
                columns = evaluation_cols
            )
            evaluation_plot_df = evaluation_plot_df.append(
                this_mod_df, 
                ignore_index=True
            )
    evaluation_plot_df.instance = evaluation_plot_df.instance.astype(int)
    
    sub_df = evaluation_plot_df[ evaluation_plot_df.Classification != 'Standard']
    # sub_df.instance = sub_df.instance.astype(int)
    all_markers = {
        'True + Champion' : 'D',
        'True' : 'X',
        'Champion' : 'D'
    }
    msize = 10
    marker_sizes = {
        'True + Champion' : msize,
        'True' : msize,
        'Champion' : msize
    }
    all_colours = {
        'True + Champion' : 'darkgreen',
        'True' : 'navy',
        'Champion' : 'darkorange'
    }
    unique_classifications = sub_df.Classification.unique()

    # Plot evaluation(s)
    num_plots = (
        include_median_likelihood 
        + include_log_likelihood
        + include_champion_percentiles
    )
    n_plots = 0 
    fig = plt.figure(
        figsize=(17, 9),
        tight_layout=True
    )
    gs = GridSpec(
        num_plots,
        1,
    )
    if include_log_likelihood:
        ax1 = fig.add_subplot(gs[0, 0])
        n_plots += 1
        sns.boxplot(
            y = 'log_likelihood', 
            x = 'instance', 
            data = evaluation_plot_df,
            ax = ax1,
            color='lightblue',
            showfliers=True
        )
        # using swarm plot since it needs to be categorical to share axis correctly with boxplot
        sns.swarmplot(
            y = 'log_likelihood', 
            x = 'instance', 
            data = sub_df, 
            ax = ax1,
            size = msize,
            hue = 'Classification',
            palette = {
                c : all_colours[c] 
                for c in unique_classifications
            },
        )


        # sns.scatterplot(
        #     y = 'log_likelihood', 
        #     x = 'instance', 
        #     data = sub_df, 
        #     # data = evaluation_plot_df[ evaluation_plot_df.Classification != 'Standard'], 
        #     ax = ax1,
        #     style='Classification',
        #     markers={
        #         c : all_markers[c]
        #         for c in unique_classifications
        #     },
        #     s = msize,
        #     hue = 'Classification',
        #     palette = {
        #         c : all_colours[c] 
        #         for c in unique_classifications
        #     },
        # )
        ax1.set_ylabel('Log likelihood')
        ax1.set_xlabel('Instance')
        if len(sub_df.instance.unique()) > 40:
            # don't list all instance IDs if too many to view easily
            ax1.set_xticks(
                np.arange(0, max(sub_df.instance.unique()) , 5)
            )
        ax1.legend()    
        ax1.set_title('Model log likelihoods')
    if include_median_likelihood:
        # median likelihoods
        ax2 = fig.add_subplot(gs[n_plots, 0])
        n_plots += 1
        sns.boxplot(
            y = 'median_likelihood', 
            x = 'instance', 
            data = evaluation_plot_df,
            ax = ax2,
            color='lightblue',
            showfliers=False
        )
        sns.scatterplot(
            y = 'median_likelihood', 
            x = 'instance', 
            data = sub_df, 
            ax = ax2,
            style='Classification',
            markers={
                c : all_markers[c]
                for c in unique_classifications
            },
            s = msize,
            hue = 'Classification',
            palette = {
                c : all_colours[c] 
                for c in unique_classifications
            }
        )
        ax2.set_ylim(0,1)
        # ax2.set_xticks([])
        ax2.set_ylabel('Median likelihood')
        ax2.set_xlabel('Instance')
        ax2.set_title('Model likelihoods (median)')


    if include_champion_percentiles:
        # plot evaluation percentileof champion models
        ax = fig.add_subplot(gs[n_plots, 0])
        n_plots += 1
        ax.set_ylabel('Percentile of champion')
        ax.set_xlabel('Number Champions')
        ax.set_title('(True) Champion models percentile log likelihood')

        
        true_champ_df = evaluation_plot_df[
            (evaluation_plot_df.Classification=='True + Champion') 
        ]
        true_df = evaluation_plot_df[
            (evaluation_plot_df.Classification=='True') 
        ]
        champ_df = evaluation_plot_df[
            (evaluation_plot_df.Classification=='Champion') 
        ]
        percentiles = np.arange(0,101,5)
        true_champ_perc = corresponding_percentile_frequencies(
            subset_to_inspect = true_champ_df,
            percentiles = percentiles
        )
        true_perc = corresponding_percentile_frequencies(
            subset_to_inspect = true_df,
            percentiles = percentiles
        )
        champ_perc = corresponding_percentile_frequencies(
            subset_to_inspect = champ_df,
            percentiles = percentiles
        )
        # horizontal bar plot
        barheight = 4
        ax.barh(
            percentiles, 
            true_champ_perc,
            height=barheight,
            label='True + Champion',
            color=all_colours['True + Champion']
        )

        ax.barh(
            percentiles, 
            champ_perc, 
            height=barheight,
            label='Champion', 
            color=all_colours['Champion'],
            left = true_champ_perc
        )

        ax.barh(
            percentiles, 
            true_perc,
            height=barheight,
            label='True',
            color=all_colours['True'],
            left = [sum(x) for x in zip(champ_perc, true_champ_perc)]
        )
        ax.axhline(
            50, ls='--', color='black', label='Median',alpha=0.3
        )
        ax.legend()        

        
    if save_directory is not None: 
        plt.savefig(
            os.path.join(
                save_directory, 
                'evaluation_by_likelihoods.png'
            )
        )

        # save the df for manual analysis afterwards
        if save_data_as_csv:
            evaluation_plot_df.to_csv(
                os.path.join(
                    save_directory, 
                    'data_evaluation_plot.csv'
                )
            )
            sub_df.to_csv(
                os.path.join(
                    save_directory, 
                    'data_classified_instances.csv'
                )
            )
        
    # return evaluation_plot_df

def round_nearest(x,a):
    return round(round(x/a)*a ,2)

def corresponding_percentile_frequencies(subset_to_inspect, percentiles):
    counted_percentiles = dict(
        subset_to_inspect.likelihood_percentile.value_counts()
    )
    percentile_freqs = [
        counted_percentiles[p]
        if p in counted_percentiles else 0
        for p in percentiles
    ]
    return percentile_freqs
    

def count_term_occurences(
    combined_results, 
    save_directory=None
):
    all_constituents = []
    all_true_constituents = []
    correct_term_counter = []

    term_counter = {}
    for i in list(combined_results.index):
        res = combined_results.iloc[i]

        constituents = eval(res['ConstituentTerms'])
        all_constituents.extend(constituents)

        true_constituents = eval(res['TrueModelConstituentTerms'])
        all_true_constituents.extend(true_constituents)

        num_correct = len(set(constituents).intersection(set(true_constituents)))
        correct_term_counter.append(num_correct)

        for term in constituents: 
            term_in_true_model = term in true_constituents
            try:
                term_counter[term]['correct'] += bool(term_in_true_model)
                term_counter[term]['incorrect'] += bool(not(term_in_true_model))
                term_counter[term]['occurences'] += 1
            except:
                term_counter[term] = {}
                term_counter[term]['correct'] = bool(term_in_true_model)
                term_counter[term]['incorrect'] = bool(not(term_in_true_model))
                term_counter[term]['occurences'] = 1

    all_true_constituents = sorted(list(set(all_true_constituents)))
    found_untrue_constituents = sorted( set(all_constituents) - set(all_true_constituents) )
    terms_ordered_by_true_presence = all_true_constituents + found_untrue_constituents
    terms_ordered_by_true_presence.reverse()

    # correct = [term_counter[term]['correct'] for term in terms_ordered_by_true_presence]
    # incorrect = [term_counter[term]['incorrect'] for term in terms_ordered_by_true_presence]
    correct = [
        term_counter[term]['correct'] 
        if (term in term_counter) else 0
        for term in terms_ordered_by_true_presence    
    ]
    incorrect = [
        term_counter[term]['incorrect'] 
        if (term in term_counter) else 0
        for term in terms_ordered_by_true_presence
    ]


    fig, ax  = plt.subplots()

    ax.barh(
        terms_ordered_by_true_presence,
        correct,
        color='g',
        left = None,
        label='Correctly'
    )

    ax.barh(
        terms_ordered_by_true_presence,
        incorrect,
        color='b',
        left = correct,
        label='Incorrectly'
    )
    ax.legend(
        title='Identified:'
    )
    max_x = max([term_counter[t]['occurences'] for t in term_counter])
    ax.set_xlim(0, max_x+1)
    ax.set_xticks(range(0, max_x+1))
    
    if save_directory is not None:
        plt.savefig(
            os.path.join(
                save_directory, 
                'term_occurences.png'
            )
        )

def inspect_times_on_nodes(combined_results, save_directory=None):
    node_ids = []
    process_ids = []
    for i in combined_results['Host']: 
        if i.startswith('node'):
            node, process = i.replace('node', '').split('-')
            node_ids.append(node)
            process_ids.append(process)
        else:
            node_ids.append(str(-1))
            process_ids.append(str(-1))

    combined_results['Node'] = node_ids
    combined_results['Process'] = process_ids

    fig, ax = plt.subplots()
    
    sns.boxplot(
        x = 'Node', 
        y = 'Time', 
        data = combined_results,
        # hue='Process',
        ax = ax, 
    )
    sns.swarmplot(
        x = 'Node', 
        y = 'Time', 
        data = combined_results,
        # hue='Process',
        color='grey',
        ax = ax, 
    )
    # ax.semilogy()
    
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time of instances on each node')
    
    if save_directory is not None:
        plt.savefig(
            os.path.join(
                save_directory, 
                'times_v_node.png'
            )
        )