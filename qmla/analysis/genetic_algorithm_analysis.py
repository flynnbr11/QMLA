import sys
import os
import numpy as np
import pandas as pd
import pickle
import itertools
import random
import copy
import scipy
import time

import sklearn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns

import qmla.utilities

def round_nearest(x,a):
    return round(round(x/a)*a ,2)

def flatten(l): 
    return [item for sublist in l for item in sublist]



def fitness_comparison(results_by_fscore, ax=None, save_directory=None):
    if ax is None: 
        fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(
        x ='f_score',
        y='fitness',
        data=results_by_fscore[ 
            (results_by_fscore['fitness_type'] !='log_likelihood') 
            & (results_by_fscore['fitness_type']!='elo_rating_raw')
        ],
        hue='fitness_type',
        ax = ax
    )
    ax.set_title("Comparison of fitness functions")
    if save_directory is not None: 
        plt.savefig(
            os.path.join(save_directory, 'genetic_fitness.png')
        )
    
    
def elo_rating_by_fscore(results_by_fscore, ax=None, save_directory=None):
    if ax is None: 
        fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(
        x ='f_score',
        y='fitness',
        data=results_by_fscore[ results_by_fscore['fitness_type']=='elo_ratings'],
#         hue='fitness_type',
        ax = ax
    )
    ax.set_title("Elo rating raw values")
    if save_directory is not None: 
        plt.savefig(
            os.path.join(save_directory, 'genetic_elo_ratings.png')
        )

def log_likelihood_by_fscore(results_by_fscore, ax=None, save_directory=None):
    if ax is None: 
        fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(
        x ='f_score',
        y='fitness',
        data=results_by_fscore[ results_by_fscore['fitness_type']=='log_likelihoods'],
        ax = ax
    )
    ax.set_title("Log likelihood")
    if save_directory is not None: 
        plt.savefig(
            os.path.join(save_directory, 'genetic_log_likelihood.png')
        )
    
def genetic_alg_num_models(results_by_fscore, ax=None, save_directory=None):
    if ax is None: 
        fig, ax = plt.subplots(figsize=(15, 6))
    sns.distplot(
        results_by_fscore['f_score'],
        bins = np.arange(0,1.01,0.05),
        kde=False,
        ax = ax, 
    )
    ax.set_title("Histogram of models by f-score")
    if save_directory is not None: 
        plt.savefig(
            os.path.join(save_directory, 'genetic_num_mods.png')
        )
    


def genetic_alg_fitness_plots(
    # results_path, 
    combined_datasets_directory,
    save_directory=None
):
    # combined_results = pd.read_csv(results_path)
    # results_by_fscore = get_f_score_dataframe(
    #     combined_results 
    # )
    results_by_fscore = pd.read_csv(
        os.path.join(combined_datasets_directory, 'fitness_df.csv')
    )    

    fig = plt.figure(
        figsize=(15, 12),
        # constrained_layout=True,
        tight_layout=True
    )
    gs = GridSpec(
        4,
        1,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])


    fitness_comparison(
        results_by_fscore, 
        ax = ax1, 
    )
    elo_rating_by_fscore(
        results_by_fscore, 
        ax = ax2, 
    )
    log_likelihood_by_fscore(
        results_by_fscore, 
        ax = ax3, 
    )
    genetic_alg_num_models(
        results_by_fscore, 
        ax = ax4, 
    )


    if save_directory is not None: 
        plt.savefig(
            os.path.join(
                save_directory, 
                'genetic_alg.png'
            )

        )
    
    

def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def f_score_from_chromosome_string(
    chromosome, 
    target_chromosome
):
    mod = np.array([int(a) for a in list(chromosome)])
    target_chromosome = np.array([int(a) for a in list(target_chromosome)])
    return sklearn.metrics.f1_score(
        mod, 
        target_chromosome
    )

def colour_by_hamming_dist(h, cmap):
    if h == 0:
        cmap_val = cmap(0.)
        alpha = 1        
    elif h == 1: 
        cmap_val = cmap(0.33)
        alpha = 0.7        
        return cmap(0.33)
    elif h <=3:
        cmap_val = cmap(0.66)
        alpha = 0.5        
    else:
        cmap_val = cmap(0.99)
        alpha = 0.5   
    alpha = 1

    return (cmap_val[0], cmap_val[1], cmap_val[2], alpha)

def map_to_full_chromosome(c, num_terms):
    return format( int( str(c), 2),  '0{}b'.format(num_terms))

def model_generation_probability(
    combined_results, 
    unique_chromosomes, 
    save_directory=None
):  
    num_terms = unique_chromosomes.num_terms.unique()[0]
    unique_chromosomes['full_chromosome'] = [map_to_full_chromosome(c, num_terms) for c in unique_chromosomes.chromosome ]   
    num_experiments = combined_results['NumExperiments'][0]
    num_particles = combined_results['NumParticles'][0]
    true_chromosome = map_to_full_chromosome(unique_chromosomes.true_chromosome[0], num_terms = num_terms)
    full_chromosome = 2**num_terms
    cmap = plt.cm.viridis
    num_runs = len(unique_chromosomes.qmla_id.unique())
    num_models = 2**num_terms
    all_models = range(num_models)

    # get prob of generation at random for all available models 
    avg_num_mods_per_instance = combined_results.NumModels.median()
    std_dev_num_mods_per_instance = combined_results.NumModels.std()
    if np.isnan(std_dev_num_mods_per_instance): 
        std_dev_num_mods_per_instance = 0

    counts = np.zeros(2**num_terms)
    num_trials = int(1e4)
    for i in range(num_trials):
        # randomly choose a number of models to sample
        num_samples = abs(int(np.random.normal(
            avg_num_mods_per_instance, 
            std_dev_num_mods_per_instance)
        ))
        try:
            model_ids = random.sample(range(2**num_terms), num_samples)
        except:
            print("Failed to draw {} samples from {} dist".format(num_samples, 2**num_terms))
            print("avg num mods per inst:", avg_num_mods_per_instance)
            print("std mods per inst:", std_dev_num_mods_per_instance)
            raise
        for m in model_ids: 
            counts[m] += 1
    counts /= num_trials
    random_sampling_prob = np.round(np.median(counts), 3)
    random_sampling_width = np.round(np.std(counts), 3)
    random_width_array_upper = [random_sampling_prob + random_sampling_width] * num_models
    random_width_array_lower = [random_sampling_prob - random_sampling_width] * num_models

    chromosomes = list(unique_chromosomes.full_chromosome)
    unique_chromosome_list = list(set(chromosomes))
    numeric_chromosomes = list(unique_chromosomes.numeric_chromosome)
    unique_numeric_chromosome_list = list(unique_chromosomes.numeric_chromosome.unique())
    f_scores = list(unique_chromosomes.f_score)

    counts = [numeric_chromosomes.count(a) for a in unique_numeric_chromosome_list]
    counts = [c/num_runs for c in counts] # so this reflects 'probability' of being generated

    array_counts = np.zeros(num_models)
    array_counts[ np.array(unique_numeric_chromosome_list)] = counts

    f_score_values = [np.round(i,2) for i in list(np.arange(0,1.001,0.01))]
    f_occurences = {f : [] for f in f_score_values}
    f_num_mods = {f : 0  for f in f_score_values}
    f_mod_present = {f : 0 for f in f_score_values}
    f_count_ratio = {f : [] for f in f_score_values}

    f_counts = {}
    f_plot_colours =  {f : [] for f in f_score_values}



    f_v_hamming = []
    hamming_v_f = []

    colours = [cmap(1)]*num_models
    for mod in all_models:
        chromosome = bin(mod)[2:].zfill(num_terms)
        h = hamming_distance(chromosome, true_chromosome)
        c = colour_by_hamming_dist(h, cmap=cmap)
        colours[mod] = c
        f_score = f_score_from_chromosome_string(chromosome = chromosome, target_chromosome=true_chromosome)
        f_score = qmla.utilities.round_nearest(f_score, 0.01)

        occurences = chromosomes.count(mod)
        f_occurences[f_score].append(occurences)
        if occurences > 0:
            f_mod_present[f_score] += 1
        f_num_mods[f_score] += 1
        f_count_ratio[f_score].append( occurences/num_runs )
        f_plot_colours[f_score].append(c)
        f_v_hamming.append(f_score)
        hamming_v_f.append(h)
    
    
    colours = [cmap(1)]*num_models
    for mod in all_models:
        chromosome = bin(mod)[2:].zfill(num_terms)
        h = hamming_distance(chromosome, true_chromosome)
        c = colour_by_hamming_dist(h, cmap=cmap)
        colours[mod] = c
        f_score = f_score_from_chromosome_string(chromosome = chromosome, target_chromosome=true_chromosome)
        f_score = qmla.utilities.round_nearest(f_score, 0.01)

        occurences = chromosomes.count(mod)
        f_occurences[f_score].append(occurences)
        if occurences > 0:
            f_mod_present[f_score] += 1
        f_num_mods[f_score] += 1
        f_count_ratio[f_score].append( occurences/num_runs )
        f_plot_colours[f_score].append(c)
        f_v_hamming.append(f_score)
        hamming_v_f.append(h)

    colours = np.array(colours)

    fig, ax = plt.subplots(figsize=(17, 7))
    ax.scatter(
        all_models, 
        array_counts, 
        edgecolors = colours,
        facecolor='none'
    )    

    label_fontsize = 20
    ax.set_xlabel('Model ID (binary representation)', fontsize=label_fontsize)
    ax.set_ylabel('Prob. of generation', fontsize=label_fontsize)
    ax.set_ylim(-0.1, 1.2)
    probs_to_label = [0, 0.25, 0.5, 0.75, 1]
    ax.set_yticks(
        probs_to_label,
    )
    ax.set_yticklabels(
        labels = probs_to_label,
        fontdict={'fontsize' : label_fontsize}
    )
    ax.axvline(
        int(true_chromosome,2),
        c = cmap(0.0), ls=':'
    )
    ax.axhline(
        random_sampling_prob, 
        label='Random generation ({}%)'.format(random_sampling_prob),
        c ='black',
        ls = '--'
    )
    handles, labels = ax.get_legend_handles_labels()

    custom_lines = [
        Line2D([0], [0], color=cmap(0.99), lw=4),
        Line2D([0], [0], color=cmap(.66), lw=4),
        Line2D([0], [0], color=cmap(0.33), lw=4),
        Line2D([0], [0], color=cmap(0.), lw=4),
    ]
    custom_labels = [
        '>3 terms wrong',
        '2-3 terms wrong', 
        '1 term wrong', 
        'Correct', 
    ]
    handles.extend(custom_lines)
    labels.extend(custom_labels)

    ax.legend(
        # custom_lines, 
        # custom_labels,
        handles, 
        labels,
        prop={'size' : 0.65*label_fontsize},
        loc='upper center',
        ncol=5
    )
    ax.set_title(
        "True chromosome: {} ({})".format(true_chromosome, int(true_chromosome,2)),
        fontsize = label_fontsize
    )
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(
        np.linspace(0,1,num_runs+1),
    )
    ax2.set_yticklabels(
        range(num_runs+1),
        fontdict={'fontsize' : label_fontsize}
    )
    if num_runs > 9: 
        ax2.set_yticks(
            [0, 0.5, 1.0],
        )
        ax2.set_yticklabels(
            ['0', str(int(num_runs/2)), str(num_runs)],
            fontdict={'fontsize' : label_fontsize}
        )

    ax2.set_ylabel('Number of occurences', fontsize=label_fontsize)
    if save_directory is not None: 
        plt.savefig(os.path.join(save_directory, 'prob_model_generation.png'))

    #############
    # f score model probability plot
    #############

    # TODO plot probability of f-score averaged
    # i.e.  in 10 runs, if 2 models have f-score 0.8, 
    # one which occurs 5 times and the other 3 times, 
    # the total probability of that f-score is 8/20 = 40%
    # so just average
    # TODO use jittering with stripplot or countsplot
    # https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
    plt.clf()
    fig, ax = plt.subplots(figsize=(17, 7))

    f_vals = []
    count_vals = []
    f_colours = []
    for f in f_score_values:
        counts = f_count_ratio[f]
        colours = f_plot_colours[f]
        for colour, count in zip(colours, counts): 
            f_vals.append(f)
            count_vals.append(count)
            f_colours.append(colour)

    ax.scatter(f_vals, count_vals, c=f_colours)
    ax.axhline(
        random_sampling_prob, 
        label='Prob random generation',
        c ='black',
        ls = '--'
    )
    ax.set_ylim(-0.1,1.2)    
    ax.set_yticks(probs_to_label)
    ax.legend(
        handles, 
        labels, 
        # custom_lines, 
        # custom_labels,
        prop={'size' : 0.65*label_fontsize},
        loc='upper center',
        ncol=5
    )

    ax.set_xlabel('Model F-score', fontsize=label_fontsize)
    ax.set_ylabel('Prob. of generation', fontsize=label_fontsize)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(
        np.linspace(0,1,num_runs+1),
    )
    ax2.set_yticklabels(
        range(num_runs+1),
        fontdict={'fontsize' : label_fontsize}
    )
    if num_runs > 9: 
        ax2.set_yticks(
            [0, 0.5, 1.0],
        )
        ax2.set_yticklabels(
            ['0', str(int(num_runs/2)), str(num_runs)],
            fontdict={'fontsize' : label_fontsize}
        )
    ax2.set_ylabel('Number of occurences', fontsize=label_fontsize)
    ax2.set_title(
        "Probability of model generation. {} experiments; {} particles; {} runs".format(
            num_experiments, 
            num_particles, 
            num_runs
        ),
        fontsize = label_fontsize
    )
    if save_directory is not None: 
        plt.savefig(os.path.join(save_directory, 'prob_f_score_generation.png'))


def correlation_fitness_f_score(
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
    fig = sns.catplot(
        y  = 'Correlation', 
        x = 'Method',
        data = fitness_correlations,
        kind='box'
    )
    fig.savefig(
        os.path.join(
            save_directory, "fitness_f_score_correlations.png"
        )
    )

