import sys
import os
import numpy as np
import pickle
import pandas as pd
import time
import seaborn as sns
import scipy
import sklearn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p = os.path.abspath(os.path.realpath(__file__))
elements = p.split('/')[:-3]
qmla_root = os.path.abspath('/'.join(elements))
sys.path.append(qmla_root)
import qmla


def run_genetic_algorithm(configuration):
    try:
        ga = qmla.growth_rules.genetic_algorithm.GeneticAlgorithmFullyConnectedLikewisePauliTerms(
            num_sites = configuration['number_sites'],
            base_terms = ['z'],
            mutation_method = configuration['mutation_method'], 
            selection_method = configuration['selection_method'],
            crossover_method = configuration['crossover_method'],
            mutation_probability = configuration['mutation_probability'], 
            unchanged_elite_num_generations_cutoff = configuration['unchanged_elite_num_generations_cutoff'], 
            log_file = configuration['log_file']
        )
        new_models = ga.random_initial_models(
            num_models = configuration['starting_population_size']
        )

        # run genetic algorithm
        for generation in range(configuration['number_generations']):
            model_f_scores = {
                mod : ga.model_f_score(mod) 
                for mod in new_models
            }

            new_models = ga.genetic_algorithm_step(
                model_fitnesses = model_f_scores
            )
            if ga.best_model_unchanged:
                print("Best model unchaged; terminating early.", flush=True)
                break
            else:
                print("Best model has chaged.", flush=True)
        print("Finished generations.", flush=True)

        champion = ga.models_ranked_by_fitness[ max(ga.models_ranked_by_fitness)][0]
        champ_f_score = ga.model_f_score(champion)

        configuration['champion_f_score'] = champ_f_score
        configuration['number_terms'] = ga.num_terms
        configuration['number_possible_models'] = 2**ga.num_terms
        print("Result:", configuration, flush=True)
        return configuration
    except:
        print("Job failed.", flush=True)
        return None


def get_all_configurations(
    log_file=None,
):
    # set up hyper parameters to sweep over
    test_setup = True
    if test_setup: 
        print("Getting reduced set of configurations to test.")
        number_of_iterations = 2
        numbers_of_sites = [5]
        numbers_of_generations = [16, ]
        starting_populations = [8, ]
        elite_models_protected = [1, ]
        mutation_probabilities = [0.1, ]
        unchanged_gen_count_for_termination = [3]        
        selection_methods = ['roulette']
        mutation_methods = ['element_wise']
        crossover_methods = ['one_point']
    else:
        # full sets to use
        print("Getting complete set of configurations to test.")
        number_of_iterations = 25
        numbers_of_sites = [5,]
        numbers_of_generations = [4, 8, 16, 32]
        starting_populations = [4, 8, 16, 32]
        elite_models_protected = [0, 1, 2, 3]
        mutation_probabilities = [0, 0.1, 0.25, 0.33]
        unchanged_gen_count_for_termination = [3, 6, 10],
        selection_methods = ['roulette']
        mutation_methods = ['element_wise']
        crossover_methods = ['one_point']


    # generate the configurations to cycle through
    config_id = 0
    all_configurations = []
    configuration_df = pd.DataFrame()
    for g in numbers_of_generations:
        for s in numbers_of_sites:
            for p in starting_populations: 
                for m in mutation_probabilities: 
                    for e in elite_models_protected: 
                        for u in unchanged_gen_count_for_termination:
                            for sel_meth in selection_methods: 
                                for mut_meth in mutation_methods: 
                                    for cross_meth in crossover_methods:         
                                        config_id += 1
                                        # resources needed, i.e. scale of how many Hamiltonian exponentiations required
                                        resources = g*(p + scipy.special.comb(p, 2)) 
                                        config = {
                                            'number_generations' : g,
                                            'number_sites' : s,
                                            'starting_population_size' : p, 
                                            'mutation_probability' : m,
                                            'selection_method' : sel_meth, 
                                            'mutation_method' : mut_meth, 
                                            'crossover_method' : cross_meth,
                                            'num_protected_elite_models' : e,
                                            'unchanged_elite_num_generations_cutoff' : u, 
                                            'config_id' : config_id, 
                                            'resources' : resources,
                                            'log_file' : log_file
                                        }
                                        all_configurations.append(config)
                                        configuration_df = configuration_df.append(
                                            pd.Series(config), 
                                            ignore_index=True
                                        )
    
    all_configurations = all_configurations * number_of_iterations
    
    return all_configurations, configuration_df
    


def plot_configuration_sweep(results, save_to_file=None):
    import matplotlib

    colours = {
        'low' : matplotlib.colors.BASE_COLORS['g'],
        'medium-low': matplotlib.colors.CSS4_COLORS['gold'], 
        'medium-high': matplotlib.colors.CSS4_COLORS['goldenrod'], 
        'high' : matplotlib.colors.BASE_COLORS['r']
    }

    fig, ax  = plt.subplots(figsize=(15, 10))

    sns.swarmplot(
        x = 'config_id',
        y = 'champion_f_score',
        data = results,
        color='grey', 
        size = 200/len(results),
        dodge=False, 
        ax = ax
    )
    sns.boxplot(
        x = 'config_id',
        y = 'champion_f_score',
        data = results,
        color='grey',
        hue='ResourceRequirement',
        hue_order = ['low', 'medium-low', 'medium-high', 'high'],
        palette = colours,
        dodge=False, 
        ax = ax
    )

    ax.set_xlabel('Configuration ID')
    ax.set_ylabel('Champion F-scores')
    
    if save_to_file is not None:
        plt.savefig(save_to_file)


def analyse_results(
    ga_results_df, 
    configuration_df, 
    result_directory
):
    # Add some stuff to complete results DF
    ga_results_df['relative_resources'] = ga_results_df.resources / ga_results_df.resources.max()
    ga_results_df['ResourceRequirement'] = [
        'low' if r <= 0.25
        else 'medium-low' if (0.25<=r) and (r<0.5)
        else 'medium-high' if (0.5<=r) and (r<0.75)
        else 'high'
        for r in ga_results_df.relative_resources
    ]
    ga_results_df['true_found'] = [True if r == 1 else False for r in ga_results_df['champion_f_score']]

    # Store complete results df
    path_to_store_result = os.path.join(
        result_directory, 
        'results.csv'
    )
    ga_results_df.to_csv( path_to_store_result )

    # Add to configuration DF and store it
    configuration_df.set_index('config_id', inplace=True)

    resources_reqd = dict(ga_results_df.groupby(['config_id']).median()['relative_resources'])
    configs_win_rate = dict(ga_results_df.groupby(['config_id']).sum()['true_found'] / ga_results_df.groupby(['config_id']).count()['true_found'])
    configs_median_f_scores = dict(ga_results_df.groupby(['config_id']).median()['champion_f_score'])
    configs_mean_f_scores = dict(ga_results_df.groupby(['config_id']).mean()['champion_f_score'])

    ordered_configs = sorted(
        configs_win_rate,
        key=configs_win_rate.get,
        reverse=True
    )

    configuration_df['win_rate'] = pd.Series(configs_win_rate)
    configuration_df['mean_f_score'] = pd.Series(configs_mean_f_scores)
    configuration_df['median_f_score'] = pd.Series(configs_median_f_scores)
    configuration_df['resources_reqd'] = pd.Series(resources_reqd)

    # configuration_df['cost_per_win'] = configuration_df['win_rate'] / configuration_df['resources_reqd']
    # configuration_df['f_value'] = configuration_df['median_f_score'] / configuration_df['resources_reqd']
    configuration_df['cost_per_win'] = configuration_df['resources'] / configuration_df['win_rate']
    configuration_df['win_per_resourrce'] = configuration_df['win_rate'] / configuration_df['resources']
    configuration_df['f_value'] = configuration_df['median_f_score'] / configuration_df['resources']

    path_to_store_configs = os.path.join(
        result_directory, 
        'configurations.csv'
    )
    configuration_df.to_csv(path_to_store_configs)


    ranked_configurations = configuration_df.sort_values(by='win_per_resourrce', ascending=False)
    summary_file = os.path.join(result_directory, 'summary.txt')
    ranked_configurations.to_string(
        buf=open(summary_file, 'w'),
        columns = [
        'f_value', 'win_per_resourrce',
        'number_generations', 'starting_population_size', 
        'resources', 'median_f_score', 'mean_f_score', 'win_rate',
        'mutation_probability', 'num_protected_elite_models',
        'mutation_method', 'selection_method', 'crossover_method', 
        ]
    )

    # Plot configurations' f scores
    plot_configuration_sweep(
        results = ga_results_df, 
        save_to_file = os.path.join(
            result_directory, 
            'param_sweep.png'
        )
    )