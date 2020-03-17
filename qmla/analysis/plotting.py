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

from qmla.analysis.analysis_and_plot_functions import fill_between_sigmas
import qmla.get_growth_rule as get_growth_rule
import qmla.model_naming as model_naming
import qmla.database_framework as database_framework
plt.switch_backend('agg')

def flatten(l): 
    # flatten list of lists
    return [item for sublist in l for item in sublist]



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

