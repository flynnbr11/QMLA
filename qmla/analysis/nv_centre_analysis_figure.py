import sys
import os
import numpy as np
import pickle
import pandas as pd
import itertools
import copy

import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
import qinfer as qi
from scipy import linalg
from matplotlib.gridspec import GridSpec
from matplotlib import rc
from matplotlib import rcParams
import math

##############
# Configure
##############

fig_base_size = 10
fig_size = (fig_base_size*1.6, fig_base_size*1.1)

plot_mainfontsize = fig_base_size
legend_fontsize = plot_mainfontsize * 0.8
marker_size = fig_base_size
linewidth = fig_base_size / 5
title_padding = fig_base_size / 4



title_font = {'fontname':'Microsoft Sans Serif', 'size':str(2*plot_mainfontsize), 'color':'black', 'weight':'normal'} 
axis_font = {'fontname':'Microsoft Sans Serif', 'size': str(2*plot_mainfontsize), 'color':'black', 'weight':'normal'} 
ticks_font = {'fontname':'Microsoft Sans Serif', 'size':str(2*plot_mainfontsize)} 
legend_font = {'fontname':'Microsoft Sans Serif', 'fontsize':str(2*plot_mainfontsize)} 
caption_font = {'fontname':'Microsoft Sans Serif', 'fontsize':str(1.8*plot_mainfontsize), 'weight': 'bold'} 
inset_caption_font = {'fontname':'Microsoft Sans Serif', 'fontsize':str(1.8*plot_mainfontsize), 'weight': 'normal'} 
legend_fontsize = 1.5*plot_mainfontsize
axis_label_padding = fig_base_size

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rc('font', family = 'Microsoft Sans Serif')
rc('xtick', labelsize=plot_mainfontsize) 
rc('ytick', labelsize=plot_mainfontsize) 
rcParams['legend.title_fontsize'] = legend_fontsize
rc('legend', fontsize=legend_fontsize) 

champ_colour = 'green'
candidate_colour = 'blue'


from matplotlib.lines import Line2D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.markers as mmark

##############
# Methods
##############

def model_pool_f_scores(
    storage_instance, 
    ax, 
):
    all_models = copy.deepcopy(storage_instance.growth_rule_storage.fitness_by_f_score)
    all_models['model_type'] = 'Candidate'

    generations = storage_instance.growth_rule_storage.fitness_by_f_score.generation.unique()
    f_score_generation_champions = {}

    for g in generations:
        champ_model_id = storage_instance.branch_champions[g]
        f_score_champ = storage_instance.model_f_scores[champ_model_id]
        f_score_generation_champions[g] = f_score_champ
        all_models.loc[
            (
                (all_models['generation'] == g)
                & 
                (all_models['model_id'] == champ_model_id)
            ), 
            'model_type'
        ] = 'Champion'

    sns.swarmplot(
        x = 'generation',
        y = 'f_score',
        data = all_models[
            all_models.model_type != 'Champion'
        ],
        ax = ax, 
        s = marker_size,
        marker='X', 
        color = 'blue'
    )

    sns.swarmplot(
        x = 'generation', 
        y = 'f_score',
        data = all_models[
            all_models.model_type == 'Champion'
        ],
        marker = '*',
        color=champ_colour, 
        s = 2*marker_size,
        ax = ax,
        dodge=False, 
    )

    legend_elements = [
        Line2D([0], [0], 
            marker='X', 
            color='b', 
            linestyle='None', 
            markersize=marker_size, 
            label='Candidate'
        ),
        Line2D([0], [0], 
            marker='*', 
            color=champ_colour, 
            label='Champion',
            linestyle='None', 
            markersize=2*marker_size,
        ),
    ]

    ax.set_xlabel('Generation', **axis_font)
    ax.set_xticks(
        range(0, int(max(generations)), 5)
    )
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('F-score', **axis_font)
    ax.set_ylim(0, 1)
    ax.legend(handles=legend_elements, fontsize=legend_fontsize)
    ax.tick_params(
        axis='both', 
        labelsize=legend_fontsize,
        pad = axis_label_padding
    )



def branch_champion_dynamics(
    run_path, 
    instance_id, 
    ax,
    branches=None,
):
    true_measurements = pickle.load(
        open(
            os.path.join(
                run_path, 
                'system_measurements.p'
            ), 'rb'
        )
    )
    storage_instance = pickle.load(
        open(os.path.join(
            run_path, 'storage_{}.p'.format(instance_id)), 'rb'
        )
    )
    storage_instance.expectation_values['time_microseconds'] = storage_instance.expectation_values['time']*1e6
    
    times = sorted(true_measurements.keys())
    times_microseconds = [t*1e6 for t in times]
    # system measurements
    ax.scatter(
        times_microseconds,
#         times, 
        [true_measurements[t] for t in times],
        color='red', 
        s = 2*marker_size, 
        label = "System"
    )
    ax.plot(
        times_microseconds,
#         times, 
        [true_measurements[t] for t in times],
        color='red', 
        lw = linewidth*0.2, 
    )

    # Plot branch champions' dynamics
    colours = itertools.cycle(['blue', 'orange', 'indigo', 'green'])
    linestyles = itertools.cycle(['--', '-.'])
    
    storage_instance.expectation_values['microseconds'] = 1e6 * storage_instance.expectation_values.time
    if branches is None:
        branches = sorted(storage_instance.branch_champions.keys())
    for b in branches:
        m = storage_instance.branch_champions[b]
        df = storage_instance.expectation_values[
            storage_instance.expectation_values.model_id == m
        ]

        sns.lineplot(
#             x = 'time', 
            x = 'microseconds', 
            y = 'exp_val',
            c = next(colours), 
            data = df, 
            label = "{}".format(b), 
            ax = ax, 
            
        )

    ax.legend(
        title='Generation champions',
        fontsize=legend_fontsize,
        ncol = 5, 
        loc='bottom center'
    )
#     ax.set_xlim(0, max(times))
    ax.set_ylim(0,1)
#     ax.set_ylabel(
#         r"$ \| \langle + \| e^{-i \hat{H}t} \| + \rangle \|^2$",
#         **axis_font, 
#     )
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(
        "Exp. value",
        **axis_font, 
    )
    ax.set_xlabel(
        'Time ($\mu s$)',
        fontsize = axis_font['size'],
    )
    ax.tick_params(
        axis='both', 
        labelsize=legend_fontsize,
        pad = axis_label_padding
    )


def term_occurences(
    combined_results, 
    ax
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
        title='Identified:',
        fontsize=legend_fontsize
    )
    ax.set_xlabel("# identifications", **axis_font)
    ax.set_ylabel("Term", **axis_font,)
    max_x = max([term_counter[t]['occurences'] for t in term_counter])
    ax.set_xlim(0, max_x+1)
    ax.set_xticks(range(0, max_x+1, 5))

    ax.tick_params(
        axis='both', 
        labelsize=legend_fontsize,
        pad = axis_label_padding
    )


def model_wins_and_occurences_by_f_score(
    all_models_generated, 
    ax
):
    bins = np.arange(0, 1.06, 0.05)

    champions = all_models_generated[
        all_models_generated.champion == True
    ]

    ax.hist(
        all_models_generated.f_score, 
        bins = bins, 
        histtype='step', 
        orientation='horizontal',
        color = candidate_colour,
        label='Occurrences'
    )
    
    champ_ax = ax.twiny()
    champ_ax.hist(
        champions.f_score, 
        bins = bins, 
        histtype='stepfilled', 
        orientation='horizontal',
        color = champ_colour,
        label='Instance champions'
    )

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0,1)
    ax.set_ylabel('F-score', **axis_font)
    ax.set_xlabel('# occurences', **axis_font)
    # ax.set_xticks([
    #     range(
    #         0, ax.get_xlim()[1], 
    #         int(0.5*(10**math.floor(np.log10(ax.get_xlim()[1]))))
    #     )
    # ])
    champ_ax.set_xlabel('# champions', **axis_font)
    ax.legend(fontsize=legend_fontsize)
    # TODO legend with champion and occurences together
    ax.tick_params(
        axis='both', 
        labelsize=legend_fontsize,
        pad = axis_label_padding
    )

def nv_centre_experimental_paper_fig_3(
    run_path,
    focus_on_instance='001',
    branches_to_draw = [1], 
    storage_instance=None, 
    all_models_generated=None, 
    save_to_file=None, 
):
    # Load data
    if storage_instance is None:
        storage_instance = pickle.load(
            open(
                os.path.join(run_path, "storage_{}.p".format(focus_on_instance)),
                'rb'
            )
        )
    if all_models_generated is None:
        all_models_generated = pd.read_csv(
            os.path.join(
                run_path, 
                "combined_datasets", 
                "models_generated.csv"
        ))
    
    # Create canvas to put plots pn
    fig, axes = plt.subplots(
        figsize=fig_size,
        constrained_layout=True,
    )
    gs = GridSpec(
        nrows=3,
        ncols=3,
    )

    # PLOTS
    # F score single instance
    ax = fig.add_subplot(
        gs[0, 1:3],
    )
    model_pool_f_scores(
        storage_instance = storage_instance,
        ax = ax
    )
    ax.set_title(
        'c', 
        loc='left', 
        fontdict = caption_font,
        pad = title_padding
    )

    # Dynamics
    ax = fig.add_subplot(
        gs[1, 1:3],
    )

    branch_champion_dynamics(
        run_path = run_path, 
        instance_id = focus_on_instance,
        ax = ax,
        branches = branches_to_draw, 
    )
    ax.set_title(
        'd', 
        loc='left', 
        fontdict = caption_font,
        pad = title_padding
    )

    # Model occurences and wins
    ax = fig.add_subplot(
        gs[0, 0],
    )
    model_wins_and_occurences_by_f_score(
        all_models_generated = all_models_generated, 
        ax = ax
    )
    ax.set_title(
        'a', 
        loc='left', 
        fontdict = caption_font,
        pad = title_padding
    )

    # Term occurences
    ax = fig.add_subplot(
        gs[1:3, 0],
    )
    combined_results = pd.read_csv(
        os.path.join(run_path, 'combined_results.csv')
    )
    term_occurences(
        combined_results = combined_results, 
        ax = ax, 
    )
    ax.set_title(
        'b', 
        loc='left', 
        fontdict = caption_font,
        pad = title_padding
    )
    
    # Bath analysis
    # num spins
    ax = fig.add_subplot(
        gs[2, 1],
    )

    path_to_num_spins_analysis = '../qmla/analysis/misc_images/num_spins.png'
    path_to_num_spins_analysis = plt.imread(path_to_num_spins_analysis)

    ax.imshow(path_to_num_spins_analysis)
    ax.axis('off')
    ax.set_title(
        'e', 
        loc='left', 
        fontdict = caption_font,
        pad = title_padding
    )

    # reconstruction
    ax = fig.add_subplot(
        gs[2, 2],
    )

    path_to_nv_reconstruction = '../qmla/analysis/misc_images/nv_reconstruction.png'
    path_to_nv_reconstruction = plt.imread(path_to_nv_reconstruction)

    ax.imshow(path_to_nv_reconstruction)
    ax.axis('off')
    ax.set_title(
        'f', 
        loc='left', 
        fontdict = caption_font,
        pad = title_padding
    )
    fig.tight_layout()
    if save_to_file is not None: 
        fig.savefig(save_to_file)
    
