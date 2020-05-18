import sys
import os
import numpy as np
import pickle
import pandas as pd
import time
import seaborn as sns
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
        ga = qmla.growth_rules.genetic_algorithm.GeneticAlgorithmQMLA(
            num_sites = configuration['number_sites'],
            base_terms = ['z'],
            mutation_method = configuration['mutation_method'], 
            selection_method = configuration['selection_method'],
            crossover_method = configuration['crossover_method'],
            mutation_probability = configuration['mutation_probability'], 
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
        number_of_iterations = 10
        numbers_of_sites = [5]
        numbers_of_generations = [12]
        starting_populations = [12,]
        elite_models_protected = [1]
        mutation_probabilities = [0.2]
        selection_methods = ['roulette']
        mutation_methods = ['element_wise']
        crossover_methods = ['one_point']
    else:
        # full sets to use
        print("Getting complete set of configurations to test.")
        number_of_iterations = 5
        numbers_of_sites = [5,]
        numbers_of_generations = [4, 8, 16, 32]
        starting_populations = [4, 8, 16, 32]
        elite_models_protected = [0, 1, 2, 4]
        mutation_probabilities = [0, 0.1, 0.25]
        selection_methods = ['roulette']
        mutation_methods = ['element_wise']
        crossover_methods = ['one_point']


    # generate the configurations to cycle through
    all_configurations = []
    for g in numbers_of_generations:
        for s in numbers_of_sites:
            for p in starting_populations: 
                for m in mutation_probabilities: 
                    for e in elite_models_protected: 
                        for sel_meth in selection_methods: 
                            for mut_meth in mutation_methods: 
                                for cross_meth in crossover_methods:         
                                    config = {
                                        'number_generations' : g,
                                        'number_sites' : s,
                                        'starting_population_size' : p, 
                                        'mutation_probability' : m,
                                        'selection_method' : sel_meth, 
                                        'mutation_method' : mut_meth, 
                                        'crossover_method' : cross_meth,
                                        'num_protected_elite_models' : e,
                                        'log_file' : log_file
                                    }
                                    # if p < (2**s / 2):
                                        # don't include where starting population over half size of total pool
                                    all_configurations.append(config)
    
    all_configurations = all_configurations * number_of_iterations
    
    return all_configurations
    
