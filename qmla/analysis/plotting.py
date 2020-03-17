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

from qmla.analysis.analysis_and_plot_functions import fill_between_sigmas, cumulativeQMDTreePlot
import qmla.get_growth_rule as get_growth_rule
import qmla.model_naming as model_naming
import qmla.database_framework as database_framework
plt.switch_backend('agg')

def flatten(l): 
    # flatten list of lists
    return [item for sublist in l for item in sublist]
















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

