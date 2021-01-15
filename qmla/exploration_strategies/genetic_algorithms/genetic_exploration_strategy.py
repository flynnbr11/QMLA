import numpy as np
import itertools
import sys
import os
import random
import copy
import scipy
import time

import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

try:
    from lfig import LatexFigure
except:
    from qmla.shared_functionality.latex_figure import LatexFigure
from qmla.exploration_strategies import exploration_strategy
import qmla.shared_functionality.probe_set_generation
import qmla.construct_models

import qmla.exploration_strategies.genetic_algorithms.genetic_algorithm

__all__ = [
    'Genetic', 
    'GeneticTest',
    'GeneticAlgorithmQMLAFullyConnectedLikewisePauliTerms'
]

def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

class Genetic(
    exploration_strategy.ExplorationStrategy
):
    r"""
    Exploration Strategy where model generation is determined through a genetic algorithm.

    """

    def __init__(
        self,
        exploration_rules,
        genes,
        true_model, 
        **kwargs
    ):
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )

        self.genes = genes
        self.true_model = true_model
        self.log_print([
            "Starting genetic ES"
            # "Genes:", genes
        ])

        self.ratings_class = qmla.shared_functionality.rating_system.ModifiedEloRating(
            initial_rating=1000,
            k_const=30
        ) # for use when ranking/rating models

        self.branch_champion_selection_stratgey = 'fitness' # 'ratings'
        self.fitness_method = 'elo_rating'
        self.prune_completed_initially = True
        self.prune_complete = True
        self.fitness_by_f_score = pd.DataFrame()
        self.fitness_df = pd.DataFrame()
        self.num_sites = qmla.construct_models.get_num_qubits(self.true_model)
        self.num_probes = 50
        self.max_num_qubits = 7
        self.hypothetical_final_generation  = False

        self.qhl_models = [
            'pauliSet_1J2_zJz_d3+pauliSet_1J3_yJy_d3+pauliSet_1J3_zJz_d3+pauliSet_2J3_xJx_d3+pauliSet_2J3_zJz_d3',
            'pauliSet_1J3_yJy_d3+pauliSet_1J3_zJz_d3+pauliSet_2J3_xJx_d3+pauliSet_2J3_zJz_d3',
            'pauliSet_1J2_zJz_d3+pauliSet_1J3_zJz_d3+pauliSet_2J3_xJx_d3+pauliSet_2J3_zJz_d3',
        ]
        self.spawn_step = 0 # 1st generation's ID

        self.mutation_probability = 0.1

        if 'log_file' not in kwargs:
            kwargs['log_file'] = self.log_file
        self.genetic_algorithm = qmla.exploration_strategies.genetic_algorithms.genetic_algorithm.GeneticAlgorithmQMLA(
            genes = genes, 
            num_sites=self.num_sites,
            true_model = self.true_model,
            mutation_probability=self.mutation_probability,
            **kwargs, 
        )

        self.true_chromosome = self.genetic_algorithm.true_chromosome
        self.true_chromosome_string = self.genetic_algorithm.true_chromosome_string

        self.num_possible_models = 2**len(self.true_chromosome)

        self.max_num_probe_qubits = self.num_sites

        # default test - 32 generations x 16 starters
        self.max_spawn_depth = 24
        self.initial_num_models = 16
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
        self.model_f_scores = {}
        self.model_points_at_step = {}     
        self.generation_model_rankings = {} 
        self.models_ranked_by_fitness = {}
        self.model_fitness_by_generation = {}
        self.fitness_correlations = {}

        self.tree_completed_initially = False
        self.max_num_models_by_shape = {
            self.num_sites : (self.initial_num_models * self.max_spawn_depth)/10,
            'other': 0
        }
        self.num_processes_to_parallelise_over = self.initial_num_models

        self.max_time_to_consider = 15
        self.min_param = 0.35
        self.max_param = 0.65


        self.fitness_mechanism_names = {
            'f_score' : r"$F_1$", 
            'hamming_distance' : r"$H$", 
            'inverse_ll' : r"$g^L$", 
            'inverse_ll_sq' : r"$-\frac{1}{L^2}$", 
            'akaike_info_criterion' : r"$\frac{1}{AIC}$",
            'aic_sq' :  r"$\frac{1}{AIC^2}$",
            'aicc' :  r"$\frac{1}{AICc}$", 
            'aicc_sq' : r"$g^{A}$",
            'bayesian_info_criterion' : r"$\frac{1}{BIC}$", 
            'bic_sq' : r"$g^{B}$",
            'akaike_weight' : r"$w_{A}$", 
            'bayes_weight' : r"$w_{B}$", 
            'mean_residuals' : r"$r_{\mu}$", 
            'mean_residuals_sq' : r"$r_{\mu}^2$", 
            'rs_mean' : r"$1-\overline{r}$",
            'rs_median' : r"$1-\tilde{r}$",
            'rs_mean_sq' : r"$g^{r}$", # r"$(1-\overline{r})^2$",
            'rs_median_sq' : r"$(1-\tilde{r})^2$",
            'bf_points' : r"$g^{p}$",
            'bf_rank' : r"$g^{R}$",  
            'elo_rating' : r"$g^{E}$", 
        }
        # self.log_print([
        #     "fitness_mechanism_names:", self.fitness_mechanism_names
        # ])

    def nominate_champions(self):
        # Choose model with highest fitness on final generation
        # if self.hypothetical_final_generation:
        #     self.log_print(["Running hypothetical step to get some models"])
        #     hypothetical_models = self.genetic_algorithm.genetic_algorithm_step(
        #         model_fitnesses = self.model_fitness_by_generation[self.spawn_step], 
        #         num_pairs_to_sample = self.initial_num_models / 2 # for every pair, 2 chromosomes proposed
        #     )
        #     self.log_print(["hypothetical generation models:", hypothetical_models])

        self.champion_model = self.models_ranked_by_fitness[self.spawn_step][0]
        self.log_print([
            "Final generation:", self.spawn_step, 
            "\nModel rankings on final generation:",
            self.models_ranked_by_fitness[self.spawn_step],
            "\nChampion:", self.champion_model
        ])
        
        return [self.champion_model]

    def analyse_generation(
        self, 
        model_points,
        model_names_ids, 
        **kwargs        
    ):
        self.spawn_step += 1

        self.log_print(["Analysing generation at spawn step ", self.spawn_step])
        self.log_print(["model names ids:", model_names_ids])
        self.model_points_at_step[self.spawn_step] = model_points
        
        # model_names_ids = model_names_ids
        sum_wins = sum(list(model_points.values()))
        if sum_wins == 0:
            sum_wins = 1 # TODO hack to get over some times passing empty dict from update_branch -- find a better way
        model_ids = list(model_points.keys())


        # model rankings  by number of wins
        ranked_model_list = sorted(
            model_points,
            key=model_points.get,
            reverse=True
        )
        ranked_models_by_name = [model_names_ids[m] for m in ranked_model_list]
        self.log_print(["Ranked models:", ranked_model_list, "\n Names:", ranked_models_by_name, "\n with fitnesses:", ])

        self.generation_model_rankings[self.spawn_step] = ranked_models_by_name
        rankings = list(range(1, len(ranked_model_list) + 1))
        rankings.reverse()
        num_points = sum(rankings) # number of points to distribute
        ranking_points = list(zip(
            ranked_models_by_name, 
            [r/num_points for r in rankings]
        ))
        ranking_points = dict(ranking_points)

        # Model ratings  (Elo ratings)        
        precomputed_ratings = self.ratings_class.get_ratings(list(model_points.keys()))
        original_ratings_by_name = {
            model_names_ids[m] : precomputed_ratings[m]
            for m in model_ids
        }
        min_rating = min(original_ratings_by_name.values())
        ratings_by_name = {
            m : original_ratings_by_name[m] - min_rating
            for m in original_ratings_by_name
        }
        self.log_print(["Rating (as fraction of starting rating):\n", ratings_by_name])
        sum_ratings = np.sum(list(ratings_by_name.values()))
        model_elo_ratings = {
            m : ratings_by_name[m]/sum_ratings
            for m in ratings_by_name
        }

        # New dictionaries which can be used as fitnesses:
        model_f_scores = {'fitness_type' : 'f_score'}
        model_hamming_distances = {'fitness_type' : 'hamming_distance'}
        model_number_wins = {'fitness_type' : 'number_wins'}
        model_win_ratio = {'fitness_type' : 'win_ratio'}
        mean_residuals = {'fitness_type' : 'mean_residuals'}
        log_likelihoods = {'fitness_type' : 'log_likelihoods'}

        # Alter finished dicts also useable as fitness
        # log_likelihoods['fitness_type'] = 'log_likelihoods'
        model_elo_ratings['fitness_type'] = 'elo_ratings'
        ranking_points['fitness_type'] = 'ranking'

        # TODO don't use available_fitness_data to fill fitness_df - get from full DF
        # available_fitness_data = [
        #     model_f_scores, model_hamming_distances, 
        #     model_number_wins, model_win_ratio, 
        #     model_elo_ratings, ranking_points, 
        #     # log_likelihoods, 
        #     # mean_residuals
        # ] 

        model_instances = [
            self.tree.model_storage_instances[m] for m in model_ids
        ]
        aic_values = {
            model.model_id : model.akaike_info_criterion
            for model in model_instances
        }
        aicc_values = {
            model.model_id : model.akaike_info_criterion_c
            for model in model_instances
        }
        min_aicc = min(aicc_values.values())
        self.log_print([
            "At generation {}, AIC of models: {}".format(self.spawn_step, aic_values)
        ])

        # store info on each model for analysis
        for m in model_ids:
            # Access the model storage instance and retrieve some attributes from there
            model_storage_instance = self.tree.model_storage_instances[m]
            self.log_print([
                "Model storage instance:", model_storage_instance
            ])
            mod = model_storage_instance.model_name
            model_number_wins[mod] = model_points[m]
            hamming_dist = self.hamming_distance_model_comparison(
                test_model = mod
            ) # for fitness use 1/H
            model_hamming_distances[mod] = (self.genetic_algorithm.num_terms - hamming_dist)/self.genetic_algorithm.num_terms
            model_f_scores[mod] = np.round(self.f_score_model_comparison(
                test_model = mod), 2
            ) # TODO get from model instance
            self.model_f_scores[m] = model_f_scores[mod]
            model_win_ratio[mod] = model_number_wins[mod]/sum_wins

            # store scores for offline analysis
            this_model_fitnesses = {
                # When adding a new fitness fnc -- add a name in self.fitness_mechanism_names
                'model' : mod, 
                'model_id' : m, 
                'generation' : self.spawn_step,
                # absolute metrics (not available in real experiments)
                'f_score' : model_f_scores[mod],
                'hamming_distance' : model_hamming_distances[mod], 
                # from storage instance
                # 'eval_log_likelihood' : model_storage_instance.evaluation_log_likelihood, 
                'inverse_ll' : -1 / model_storage_instance.evaluation_log_likelihood,
                'inverse_ll_sq' : (-1 / model_storage_instance.evaluation_log_likelihood)**2,
                'akaike_info_criterion' : 1 / model_storage_instance.akaike_info_criterion, 
                'aicc' : 1 / model_storage_instance.akaike_info_criterion_c, 
                'aic_sq' : (1 / model_storage_instance.akaike_info_criterion)**2, 
                'aicc_sq' : (1 / model_storage_instance.akaike_info_criterion_c)**2, 
                'bayesian_info_criterion' : (1 / model_storage_instance.bayesian_info_criterion),
                'bic_sq' : (1 / model_storage_instance.bayesian_info_criterion)**2,
                'akaike_weight' : np.e**( (min_aicc - model_storage_instance.akaike_info_criterion_c)/2),
                'bayes_weight' : np.e**(-1*model_storage_instance.bayesian_info_criterion/2),
                'mean_residuals' : 1 - model_storage_instance.evaluation_mean_pr0_diff,
                'mean_residuals_sq' : (1 - model_storage_instance.evaluation_mean_pr0_diff)**2,
                'rs_mean' : 1 - model_storage_instance.evaluation_residual_squares['mean'],
                'rs_median' : 1 - model_storage_instance.evaluation_residual_squares['median'],
                'rs_mean_sq' : (1 - model_storage_instance.evaluation_residual_squares['mean'])**2,
                'rs_median_sq' : (1 - model_storage_instance.evaluation_residual_squares['median'])**2,
                # relative to other models in this branch
                'bf_points' : model_win_ratio[mod], 
                'bf_rank' : ranking_points[mod], 
                'elo_rating' : model_elo_ratings[mod], 
                # 'original_elo_rating' : original_ratings_by_name[mod],
            }

            self.fitness_by_f_score = (
                self.fitness_by_f_score.append(
                    pd.Series(this_model_fitnesses),
                    ignore_index=True
                )
            )

            recorded_fitness_types = list(
                this_model_fitnesses.keys()
                - ['model', 'model_id', 'generation',
                    'hamming_distance', 
                ]
            )
            for f in recorded_fitness_types:
                try:
                    new_entry = pd.Series(
                        {
                            'generation' : this_model_fitnesses['generation'],
                            'f_score' : this_model_fitnesses['f_score'], 
                            'fitness' : this_model_fitnesses[f], 
                            'fitness_type' : f,
                            'fitness_type_name' : self.fitness_mechanism_names[f],
                            'active_fitness_method' : self.fitness_method==f,
                        }
                    )
                    self.fitness_df = self.fitness_df.append(
                        new_entry, ignore_index=True)
                except:
                    self.log_print([
                        "fitness name keys:", list(self.fitness_mechanism_names.keys())
                        # "f={}; type name = {}".format(f, self.fitness_mechanism_names[f])
                    ])
                    raise

        # Extract fitness specified by user (exploration strategy's fitness_method attribute) 
        # to use for generating models within genetic algorithm
        fitnesses = self.fitness_by_f_score[
            self.fitness_by_f_score.generation == self.spawn_step
        ][ ['model', self.fitness_method] ]

        genetic_algorithm_fitnesses = dict(zip(fitnesses['model'], fitnesses[self.fitness_method]))  

        self.log_print([
            "fitness method:{} => Fitnesses={}".format(
                self.fitness_method, genetic_algorithm_fitnesses
            )
        ])
        self.models_ranked_by_fitness[self.spawn_step] = sorted(
            genetic_algorithm_fitnesses,
            key=genetic_algorithm_fitnesses.get,
            reverse=True
        )
        self.model_fitness_by_generation[self.spawn_step] = genetic_algorithm_fitnesses

        self.genetic_algorithm.consolidate_generation(
            model_fitnesses = genetic_algorithm_fitnesses
        )

        # return genetic_algorithm_fitnesses
        return self.models_ranked_by_fitness[self.spawn_step]

    def generate_models(
        self,
        model_list,
        **kwargs
    ):       
        # Analysis of the previous generation is called by the exploration strategy tree. 
        genetic_algorithm_fitnesses = self.model_fitness_by_generation[self.spawn_step]
      
        self.log_print([
            "Spawn step:", self.spawn_step,
        ])

        # Spawn models from genetic algorithm
        new_models = self.genetic_algorithm.genetic_algorithm_step(
            model_fitnesses = genetic_algorithm_fitnesses, 
            num_pairs_to_sample = self.initial_num_models / 2 # for every pair, 2 chromosomes proposed
        )

        return new_models

    def finalise_model_learning(self, **kwargs):
        return

    def hamming_distance_model_comparison(
        self, 
        test_model,
        target_model=None, 
    ):
        if target_model is None:
            target_model = self.true_chromosome_string
        else:
            target_model = self.genetic_algorithm.chromosome_string(
            self.genetic_algorithm.map_model_to_chromosome(
                target_model
        ))            
        test_model = self.genetic_algorithm.chromosome_string(
            self.genetic_algorithm.map_model_to_chromosome(
                test_model
        ))

        h = sum(c1 != c2 for c1, c2 in zip(test_model, target_model))
        return h

    def f_score_model_comparison(
        self,
        test_model,
        target_model=None, 
        beta=1,  # beta=1 for F1-score. Beta is relative importance of sensitivity to precision
    ):
        if target_model is None:
            target_model = self.true_model

        true_set = set(
            self.latex_name(mod) for mod in
            qmla.construct_models.get_constituent_names_from_name(target_model)
        )
        terms = [
            self.latex_name(
                term
            )
            for term in
            qmla.construct_models.get_constituent_names_from_name(
                test_model
            )
        ]
        learned_set = set(sorted(terms))

        total_positives = len(true_set)
        true_positives = len(true_set.intersection(learned_set))
        false_positives = len(learned_set - true_set)
        false_negatives = len(true_set - learned_set)
        precision = true_positives / \
            (true_positives + false_positives)
        sensitivity = true_positives / total_positives
        try:
            f_score = (
                (1 + beta**2) * (
                    (precision * sensitivity)
                    / (beta**2 * precision + sensitivity)
                )
            )
        except BaseException:
            # both precision and sensitivity=0 as true_positives=0
            f_score = 0
        return f_score

    def f_score_from_chromosome_string(
        self, 
        chromosome, 
    ):
        mod = np.array([int(a) for a in list(chromosome)])
        
        try:
            f =  sklearn.metrics.f1_score(
                mod, 
                self.true_chromosome
            )
            return f
        except:
            self.log_print([
                "F score from chromosome {} with mod {} not working against true chrom {}".format(
                    mod, chromosome, self.true_chromosome
                )
            ])
            raise

    def exploration_strategy_finalise(
        self
    ):        
        # hypothetical generation_models
        if self.hypothetical_final_generation:
            # TODO this will cause a crash in QHL mode since. 
            # in general this should be turned off so not worth a large fix
            self.log_print(["Running hypothetical step to get some models"])
            hypothetical_models = self.genetic_algorithm.genetic_algorithm_step(
                model_fitnesses = self.model_fitness_by_generation[self.spawn_step-1], 
                num_pairs_to_sample = self.initial_num_models / 2 # for every pair, 2 chromosomes proposed
            )
            self.log_print(["hypothetical generation models:", hypothetical_models])

        self.storage.fitness_correlations = self.fitness_correlations
        self.storage.fitness_by_f_score = self.fitness_by_f_score
        self.storage.fitness_df = self.fitness_df
        self.storage.true_model_chromosome = self.true_chromosome_string
        self.storage.ratings_df = self.ratings_class.ratings_df
        gene_pool = self.genetic_algorithm.gene_pool
        gene_pool['objective_function'] = self.fitness_mechanism_names[self.fitness_method]
        self.storage.gene_pool = gene_pool
        birth_register = self.genetic_algorithm.birth_register
        birth_register['objective_function'] = self.fitness_mechanism_names[self.fitness_method]
        birth_register['max_time_considered'] = self.max_time_to_consider
        self.storage.birth_register = birth_register
        self.storage.ratings = self.ratings_class.all_ratings

        chromosomes = sorted(list(set(
            self.genetic_algorithm.previously_considered_chromosomes)))
        self.unique_chromosomes = pd.DataFrame(
            columns=['chromosome', 'numeric_chromosome', 'f_score', 'num_terms', 'hamming_distance'])
        for c in chromosomes:
            hamming_dist = self.hamming_distance_model_comparison(
                test_model = self.genetic_algorithm.map_chromosome_to_model(c)
            ) # for fitness use 1/H

            chrom_data = pd.Series({
                'chromosome' : str(c), 
                'numeric_chromosome' : int(c, 2),
                'num_terms' : self.genetic_algorithm.num_terms, 
                'hamming_distance' : hamming_dist,
                'f_score' : np.round(self.f_score_from_chromosome_string(c), 3) 
            })
            self.unique_chromosomes.loc[len(self.unique_chromosomes)] = chrom_data
        self.log_print(["self.unique_chromosomes:\n", self.unique_chromosomes])
        self.storage.unique_chromosomes = self.unique_chromosomes

        dud_chromosome = str('1' +'0'*self.genetic_algorithm.num_terms)
        if dud_chromosome in chromosomes:
            self.log_print([
                "{} in previous chromosomes:\n{}".format(
                    dud_chromosome, 
                    self.genetic_algorithm.previously_considered_chromosomes
                )
            ])
        chromosome_numbers = sorted([int(c,2) for c in chromosomes])
        # self.exploration_strategy_specific_data_to_store['chromosomes_tested'] = chromosome_numbers
        try:
            f_scores = []
            for c in chromosomes:
                try:
                    f_scores.append(np.round(self.f_score_from_chromosome_string(c), 3) )
                except:
                    self.log_print([
                        "Could not compute f score for chromosome: {}".format(c)
                    ])
            # self.exploration_strategy_specific_data_to_store['f_score_tested_models' ] = f_scores
        except:
            self.log_print([
                "Could not compute f score for chromosome list: {}".format(chromosomes)
            ])
            pass

        self.storage.chromosomes_tested = chromosome_numbers
        self.storage.f_score_tested_models = f_scores

    def check_tree_completed(
        self,
        spawn_step,
        **kwargs
    ):
        if self.spawn_step == self.max_spawn_depth:
            self.log_print(["Terminating at spawn depth ", self.spawn_step])
            return True
        elif self.genetic_algorithm.best_model_unchanged:
            self.champion_determined = True
            self.champion_model = self.genetic_algorithm.most_elite_models_by_generation[
                self.genetic_algorithm.genetic_generation - 1
            ]

            self.log_print(
                [
                    "Terminating search early (after {} generations) b/c elite model unchanged in {} generations.".format(
                        self.genetic_algorithm.genetic_generation, 
                        self.genetic_algorithm.unchanged_elite_num_generations_cutoff
                    ),
                    "\nDeclaring champion:", self.champion_model
                ]
            )
            # check if elite model hasn't changed in last N generations
            return True
        else:
            self.log_print([
                "Elite models changed recently; continuing search."
            ])
            return False

    def check_tree_pruned(self, **kwargs):
        # no pruning for GA, winner is champion of final branch
        return True

    def exploration_strategy_specific_plots(
        self,
        true_model_id,
        qmla_id=0, 
        plot_level=2, 
        **kwargs
    ):
        self.qmla_id = qmla_id
        self.plot_level = plot_level
        self.log_print(["genetic alg plots"])
        super().exploration_strategy_specific_plots(
            **kwargs
        )

        plot_methods_by_level = {
            1 : [],
            2 : [
                self.plot_correlation_fitness_with_f_score,
                self.plot_fitness_v_fscore_by_generation,
                self._plot_gene_pool_progression,
            ], 
            3 : [
                self.plot_fitness_v_fscore,
                self.plot_fitness_v_generation,
            ], 
            4 : [
                self.plot_model_ratings,
                self.plot_gene_pool,
            ], 
            5 : [
                self.plot_generational_metrics,
                self.plot_selection_probabilities
            ], 
            6 : [], 
        }
        self.log_print([
            "Plotting methods:", plot_methods_by_level
        ])

        for pl in range(self.plot_level + 1):
            if pl in plot_methods_by_level:
                self.log_print(["Plotting for plot_level={}".format(pl)])
                for method in plot_methods_by_level[pl]:
                    try:
                        method()
                    except Exception as e:
                        self.log_print([
                            "plot failed {} with exception: {}".format(method.__name__, e)
                        ])

        # Plots that need arguments so are called individually
        if self.plot_level >= 2:
            try:
                self.ratings_class.plot_models_ratings_against_generation(
                    f_scores = self.model_f_scores, 
                    save_directory = self.save_directory,
                    f_score_cmap=self.f_score_cmap
                )
            except Exception as e:
                self.log_print([
                    "plot failed plot_models_ratings_against_generation with error ", e
                ])

            try:
                self.ratings_class.plot_rating_progress_single_model(
                    target_model_id = champion_model_id,
                    save_to_file  = os.path.join(
                        self.save_directory, 
                        "ratings_progress_champion.png"
                    )
                )
                if true_model_id != -1 and true_model_id != champion_model_id:
                    self.ratings_class.plot_rating_progress_single_model(
                        target_model_id = true_model_id,
                        save_to_file  = os.path.join(
                            save_directory, 
                            "ratings_progress_true_model.png"
                        )
                    )
            except Exception as e:
                self.log_print([
                    "plot failed plot_rating_progress_single_model with error ", e
                ])

    def plot_correlation_fitness_with_f_score(
        self,
        save_to_file=None, 
    ):
        plt.clf()
        correlations = pd.DataFrame(
            columns = ['Generation', 'Method', 'Correlation']
        )
        fitness_types_to_ignore = ['f_score', 'hamming_distance']
        for t in self.fitness_df.fitness_type.unique():
            if t not in fitness_types_to_ignore:
                this_fitness_type = self.fitness_df[
                    self.fitness_df['fitness_type'] == t
                ]
                
                for g in this_fitness_type.generation.unique():
                    this_type_this_gen = this_fitness_type[
                        this_fitness_type.generation == g
                    ]
                    
                    corr = this_type_this_gen['f_score'].corr(
                        this_type_this_gen['fitness']
                    )
                    cov = this_type_this_gen['f_score'].cov(
                        this_type_this_gen['fitness']
                    )
                    
                    corr = {
                        'Generation' : g,
                        'Method' : self.fitness_mechanism_names[t], 
                        # 'Method' : t, 
                        'Correlation' : corr,
                        'Covariance' : cov, 
                    }
                    correlations = correlations.append(
                        pd.Series(corr),
                        ignore_index=True
                    )
                
        self.fitness_correlations = correlations
        self.log_print(["fitness correlations:\n", self.fitness_correlations])
        fig, ax = plt.subplots(figsize=(15, 10))

        if len(correlations.Generation.unique()) == 1:
            sns.scatterplot(
                y = 'Correlation', 
                x = 'Generation', 
                # style= 'Method', 
                hue = 'Method',
                data = correlations,
                ax = ax,
                # markers = ['*', 'X', '<', '^'],
            )
        else:
            sns.lineplot(
                y = 'Correlation', 
                x = 'Generation', 
                # style= 'Method', 
                hue = 'Method',
                data = correlations,
                ax = ax,
                markers = ['*', 'X', '<', '^'],
            )
        ax.axhline(0, ls='--', c='k')

        if save_to_file is None:
            save_to_file = os.path.join(
                self.save_directory, 
                'correlations_bw_fitness_and_f_score.png'.format(self.qmla_id)
            )

        plt.savefig(save_to_file)



    def plot_fitness_v_generation(self, save_to_file=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.clf()
        fig, ax = plt.subplots()
        sns.set(rc={'figure.figsize':(11.7,8.27)})

        cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        sns.boxplot(
            x='generation', 
            y='fitness', 
            data = self.fitness_df[ 
                # self.fitness_df['fitness_type'] == 'model_hamming_distances' 
                self.fitness_df['active_fitness_method'] == True
            ],
            ax = ax
        )
        ax.legend(loc='lower right')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title("Fitness method: {}".format(self.fitness_method))
        # ax.set_xlim((0,1))
        if save_to_file is None:
            save_to_file = os.path.join(
                self.save_directory, 
                'fitness_v_generation.png'.format(self.qmla_id)
            )

        plt.savefig(save_to_file)


    def plot_fitness_v_fscore_by_generation(
        self, save_to_file=None
    ):
        plt.clf()
        sanity_check_df = self.fitness_df[ 
            (self.fitness_df['fitness_type'] == 'f_score') 
            | (self.fitness_df['fitness_type'] == 'model_hamming_distances') 
        ]
        candidate_fitnesses = self.fitness_df[ 
            (self.fitness_df['fitness_type'] == 'elo_rating') 
            | (self.fitness_df['fitness_type'] == 'ranking') 
            | (self.fitness_df['fitness_type'] == 'model_win_ratio') 
        ]

        g = sns.FacetGrid(
            candidate_fitnesses,
            row ='generation',
            hue='fitness_type',
            hue_kws=dict(marker=["x", "+", "*"]),
            # col_wrap=5, 
            xlim=(-0.1, 1.1), 
            # ylim=(0,1),
            size=4, 
            aspect=2
        )
        g = (
            g.map(plt.scatter,  'f_score', 'fitness').add_legend()
        )

        if save_to_file is None: 
            save_to_file = os.path.join(
                self.save_directory, 
                'fitness_types.png'.format(self.qmla_id)
            )
        plt.savefig(save_to_file)

    def plot_model_ratings(self, save_to_file=None):
        plt.clf()
        ratings = self.ratings_class.all_ratings
        generations = [int(g) for g in ratings.generation.unique()]
        num_generations = len(generations)

        lf = LatexFigure(
            use_gridspec=True, 
            gridspec_layout=(num_generations, 1)
        )

        # TODO : unique linestyle and colour combo for each model ID and tracks across subplots
        ratings['Model ID'] = ratings['model_id']

        for gen in generations:
            ax = lf.new_axis()

            this_gen_ratings = ratings[ratings.generation==gen]
            colours = {
                m : self.f_score_cmap(self.model_f_scores[m])
                for m in this_gen_ratings['model_id']
            }
            sns.lineplot(
                x = 'idx', 
                y = 'rating', 
                hue = r'Model ID', 
                hue_order = sorted(this_gen_ratings.model_id.unique()),
                data=this_gen_ratings, 
                ax = ax,
                legend='full',
                palette = colours, 
            )

            ax.set_title('Generation {}'.format(gen), pad = -15)   
            ax.set_xlabel("")
            ax.set_ylabel("Elo rating")
            ax.legend(bbox_to_anchor=(1, 1))

        if save_to_file is None: 
            save_to_file = os.path.join(
                self.save_directory, 
                'ratings.png'.format(self.qmla_id)
            )

        lf.save(save_to_file)

    def plot_fitness_v_fscore(self, save_to_file=None):
        plt.clf()
        fig, ax = plt.subplots()
        sns.set(rc={'figure.figsize':(11.7,8.27)})

        cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        sns.scatterplot(
            x='f_score', 
            y='elo_rating', 
            # hue='generation',
            # palette = cmap,
            label='Rating',
            data = self.fitness_by_f_score,
            ax = ax
        )

        sns.scatterplot(
            x='f_score', 
            y='win_ratio', 
            # hue='generation',
            # palette = cmap,
            label='Win ratio',
            data = self.fitness_by_f_score,
            ax = ax
        )

        ax.legend(loc='lower right')
        ax.set_xlabel('F score')
        ax.set_ylabel('Fitness (as probability)')
        # bplot.set_ylim((0,1))
        ax.set_xlim((-0.05,1.05))
        if save_to_file is None:
            save_to_file = os.path.join(
                self.save_directory, 
                'fitness_v_fscore.png'.format(self.qmla_id)
            )

        ax.figure.savefig(save_to_file)

    def plot_gene_pool(self, save_to_file=None):
        ga = self.genetic_algorithm

        plt.clf()
        fig, axes = plt.subplots(
            figsize=(10, 8),
            constrained_layout=True, 
        )

        gs = GridSpec(
            nrows = 2,
            ncols = 1,
            height_ratios=[7, 1]
        )
        label_fontsize = 10
        # TODO get f score cmap from exploration strategy
        # f_score_cmap = matplotlib.colors.ListedColormap(["sienna", "red", "darkorange", "gold", "blue"])
        f_score_cmap = self.f_score_cmap

        # Bar plots for probability of gene being selected, coloured by f score
        ax = fig.add_subplot(gs[0,0])

        generations = list(sorted(ga.gene_pool.generation.unique()))
        probability_grouped_by_f_by_generation = {
            g : 
                {
                    f : ga.gene_pool[ 
                        (ga.gene_pool.f_score == f)
                        & (ga.gene_pool.generation == g)
                    ].probability.sum()
                for f in ga.gene_pool.f_score.unique() 
                }
            for g in generations
        }
        probability_grouped_by_f_by_generation = pd.DataFrame(probability_grouped_by_f_by_generation).T

        sorted_f_scores = list(sorted(ga.gene_pool.f_score.unique()))
        below = [0]*len(generations)
        for f in sorted_f_scores[:]:
            probs_this_f = list(probability_grouped_by_f_by_generation[f])
            ax.bar(
                generations,
                probs_this_f,
                color = f_score_cmap(f),
                bottom = below,
                edgecolor=['black']*len(generations)
            )

            below = [b+p for b,p in zip(below,probs_this_f)]
        ax.set_xticks(generations)
        ax.set_ylabel('Probability', fontsize=label_fontsize)
        ax.set_xlabel('Generation', fontsize=label_fontsize)
        ax.set_title('Gene pool', fontsize=label_fontsize)

        # Colour bar
        ax = fig.add_subplot(gs[1,0])
        sm = plt.cm.ScalarMappable(
            cmap = f_score_cmap, 
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array(np.linspace(0, 1, 100))
        plt.colorbar(sm, cax=ax, orientation='horizontal')
        ax.set_xlabel('F-score',  fontsize=label_fontsize)

        # Save figure
        if save_to_file is None:
            save_to_file = os.path.join(
                self.save_directory, 
                'gene_pool.png'
            )

        fig.savefig(save_to_file)

    def plot_selection_probabilities(self, save_to_file=None): 
        generations = sorted(self.genetic_algorithm.gene_pool.generation.unique())
        self.log_print(["[plot_selection_probabilities] generations:", generations])
        lf = LatexFigure(auto_gridspec=len(generations))

        for g in generations:
            ax = lf.new_axis()
            this_gen_genes = self.genetic_algorithm.gene_pool[
                self.genetic_algorithm.gene_pool.generation == g
            ]
            f_scores = this_gen_genes.f_score
            colours = [self.f_score_cmap(f) for f in f_scores]
            probabilities = this_gen_genes.probability    
            
            ax.pie(
                probabilities, 
                colors = colours, 
                radius=2,
            )

        if save_to_file is None:
            save_to_file = os.path.join(
                self.save_directory, 
                'selection_probabilities.png'
            )
        lf.save(save_to_file)


    def plot_generational_metrics(self, save_to_file=None):

        fig, axes = plt.subplots(figsize=(15, 10), constrained_layout=True)
        gs = GridSpec(nrows=2, ncols = 1, )

        ax = fig.add_subplot(gs[0,0])
        sns.boxplot(
            y = 'f_score', 
            x = 'generation', 
            data = self.fitness_by_f_score,
            ax = ax
        )
        ax.set_ylabel('F-score')
        ax.set_xlabel('Generation')
        ax.set_title('F score')
        ax.set_ylim(0,1)
        ax.legend()

        ax = fig.add_subplot(gs[1,0])
        sns.boxplot(
            y = 'log_likelihood', 
            x = 'generation', 
            data = self.fitness_by_f_score,
            ax = ax
        )
        ax.set_ylabel('log-likelihood')
        ax.set_xlabel('Generation')
        ax.set_title('Evaluation log likeihood')
        ax.legend()

        # Save figure
        if save_to_file is None:
            save_to_file = os.path.join(
                self.save_directory, 
                'generation_progress.png'
            )

        fig.savefig(save_to_file)

    def _plot_gene_pool_progression(
        self, 
    ):
        lf = LatexFigure()
        ax = lf.new_axis()
        gene_pool = self.genetic_algorithm.gene_pool
        gene_pool.sort_values('f_score', inplace=True, ascending=False)

        self.gene_pool_progression(
            gene_pool = gene_pool,
            ax = ax, 
            f_score_cmap = self.f_score_cmap,
        )
        lf.save(
            save_to_file = os.path.join(
                self.save_directory, 
                'gene_pool_progression.png'
            )
        )

    @staticmethod
    def gene_pool_progression(gene_pool, ax, f_score_cmap=None, draw_cbar=True, cbar_ax=None):
        if f_score_cmap is None:
            f_score_cmap = matplotlib.cm.RdBu
        num_models_per_generation = len(gene_pool[gene_pool.generation == 1])
        num_generations = gene_pool.generation.nunique()
        f_scores_of_gene_pool = np.empty((num_models_per_generation, num_generations))
        for g in gene_pool.generation.unique():

            f_scores_by_gen = gene_pool[
                gene_pool.generation == g
            ].f_score

            f_scores_of_gene_pool[:, g-1] = f_scores_by_gen

        sns.heatmap(
            f_scores_of_gene_pool,
            cmap = f_score_cmap,
            vmin = 0, 
            vmax=1,
            ax = ax,
            cbar=draw_cbar, 
            cbar_kws = dict(
                label=r"$F_1$-score",
                aspect=25, 
                ticks=[0,0.5,1],
            )
        )
        ax.set_yticks([])
        xtick_pos = range(5, num_generations+1, 5)
        ax.set_xticks([g-0.5 for g in xtick_pos])
        ax.set_xticklabels(
            xtick_pos
        )
        ax.set_xlabel('Generation')

        if cbar_ax is not None:
            cbar = ax.collections[0].colorbar
            cbar.ax.set_ylabel(r"$F_1$", rotation=0, labelpad=10) # if F horizontal
            cbar.ax.yaxis.set_label_position("right", )
            cbar.ax.tick_params(labelleft=True, labelright=False )



class GeneticTest(
    Genetic
):
    r"""
    Exactly as the genetic exploration strategy, but small depth to test quickly.

    """

    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        true_model = 'pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_zJz_d4'
        self.true_model = qmla.construct_models.alph(true_model)
        num_sites = qmla.construct_models.get_num_qubits(true_model)
        terms = []
        for i in range(1, 1 + num_sites):
            for j in range(i + 1, 1 + num_sites):
                for t in ['x', 'y', 'z']:
                    new_term = 'pauliSet_{i}J{j}_{o}J{o}_d{N}'.format(
                        i= i, j=j, o=t, N=num_sites, 
                    )
                    terms.append(new_term)
        
        super().__init__(
            exploration_rules = exploration_rules,
            genes = terms, 
            true_model = self.true_model, 
            **kwargs
        )
        self.max_spawn_depth = 2
        self.max_num_probe_qubits = self.num_sites
        self.initial_num_models = 6
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
        self.tree_completed_initially = False
        self.max_num_models_by_shape = {
            self.num_sites : (self.initial_num_models * self.max_spawn_depth)/10,
            'other': 0
        }
        self.num_processes_to_parallelise_over = self.initial_num_models
 

class GeneticAlgorithmQMLAFullyConnectedLikewisePauliTerms(
    Genetic
):
    def __init__(
        self,
        exploration_rules,
        true_model, 
        num_sites=None, 
        base_terms=['x', 'y', 'z'],
        **kwargs
    ):
        if num_sites is None: 
            num_sites = qmla.construct_models.get_num_qubits(true_model)
        terms = []
        for i in range(1, 1 + num_sites):
            for j in range(i + 1, 1 + num_sites):
                for t in base_terms:
                    new_term = 'pauliSet_{i}J{j}_{o}J{o}_d{N}'.format(
                        i= i, j=j, o=t, N=num_sites, 
                    )
                    terms.append(new_term)
        
        super().__init__(
            exploration_rules = exploration_rules,
            genes = terms, 
            true_model = true_model, 
            **kwargs
        )



