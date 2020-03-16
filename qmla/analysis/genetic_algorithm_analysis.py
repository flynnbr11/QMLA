import sys
import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random
import copy
import scipy
import time
from matplotlib.gridspec import GridSpec
import seaborn as sns


def round_nearest(x,a):
    return round(round(x/a)*a ,2)


def get_f_score_dataframe(combined_results):

    results_by_fscore = pd.DataFrame()
    for i in combined_results.index:
        ratings_list = eval(dict(combined_results['GrowthRuleStorageData'])[i])['f_score_fitnesses']
        for result in ratings_list: 
            f_score = float(round_nearest(result[0], 0.05))
            run = i
            generation = ratings_list.index(result)
            # win ratio
            results_by_fscore = (
                results_by_fscore.append(
                    pd.Series(
                    {
                        'f_score' : f_score,
                        'fitness' : np.round(result[1], 2),
                        'fitness_type' : 'win_ratio',
                        'run' : run, 
                        'generation' : generation
                    }), 
                    ignore_index=True
                )
            )    

            # rating
            results_by_fscore = (
                results_by_fscore.append(
                    pd.Series(
                    {
                        'f_score' : float(round_nearest(result[0], 0.05)),
                        'fitness' : np.round(result[2], 2),
                        'fitness_type' : 'elo_rating',
                        'run' : run, 
                        'generation' : generation
                    }), 
                    ignore_index=True
                )
            )    

            # rating
            results_by_fscore = (
                results_by_fscore.append(
                    pd.Series(
                    {
                        'f_score' : float(round_nearest(result[0], 0.05)),
                        'fitness' : np.round(result[3], 2),
                        'fitness_type' : 'elo_rating_raw',
                        'run' : run, 
                        'generation' : generation
                    }), 
                    ignore_index=True
                )
            )    


            # ranking
            results_by_fscore = (
                results_by_fscore.append(
                    pd.Series(
                    {
                        'f_score' : float(round_nearest(result[0], 0.05)),
                        'fitness' : np.round(result[4], 2),
                        'fitness_type' : 'ranking',
                        'run' : run, 
                        'generation' : generation
                    }), 
                    ignore_index=True
                )
            )    
            
        return results_by_fscore
    

def fitness_comparison(results_by_fscore, ax=None, save_directory=None):
    if ax is None: 
        fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(
        x ='f_score',
        y='fitness',
        data=results_by_fscore[ results_by_fscore['fitness_type']!='elo_rating_raw'],
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
        data=results_by_fscore[ results_by_fscore['fitness_type']=='elo_rating_raw'],
#         hue='fitness_type',
        ax = ax
    )
    ax.set_title("Elo rating raw values")
    if save_directory is not None: 
        plt.savefig(
            os.path.join(save_directory, 'genetic_elo_ratings.png')
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
    results_path, 
    save_directory=None
):
    combined_results = pd.read_csv(results_path)
    results_by_fscore = get_f_score_dataframe(
        combined_results 
    )    

    fig = plt.figure(
        figsize=(15, 12),
        # constrained_layout=True,
        tight_layout=True
    )
    gs = GridSpec(
        3,
        1,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    fitness_comparison(
        results_by_fscore, 
        ax = ax1, 
        # save_directory
    )
    elo_rating_by_fscore(
        results_by_fscore, 
        ax = ax2, 
        # save_directory
    )
    genetic_alg_num_models(
        results_by_fscore, 
        ax = ax3, 
        # save_directory
    )

    if save_directory is not None: 
        plt.savefig(
            os.path.join(
                save_directory, 
                'genetic_alg.png'
            )

        )
    
    

