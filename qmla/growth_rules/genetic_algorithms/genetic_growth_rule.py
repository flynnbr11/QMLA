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
import matplotlib.pyplot as plt
import seaborn as sns

from qmla.growth_rules import growth_rule
import qmla.shared_functionality.probe_set_generation
import qmla.database_framework

import qmla.growth_rules.genetic_algorithms.genetic_algorithm

__all__ = [
    'Genetic', 
    'GeneticTest'
    # 'GeneticAlgorithmQMLA'
]
# flatten list of lists
def flatten(l): return [item for sublist in l for item in sublist]

def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


class Genetic(
    growth_rule.GrowthRule
):

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        # self.true_model = 'pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_2J3_yJy_d4+pauliSet_1J4_yJy_d4'
        # self.true_model = 'pauliSet_1J2_xJx_d3+pauliSet_1J2_yJy_d3+pauliSet_2J3_yJy_d3+pauliSet_2J3_zJz_d3'
        # self.ising_full_connectivity = 'pauliSet_1J2_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4'
        self.ratings_class = qmla.growth_rules.rating_system.ELORating(
            initial_rating=1500,
            k_const=30
        ) # for use when ranking/rating models
        self.fitness_method = 'f_scores' # 'elo_ratings'
        self.prune_completed_initially = True
        self.prune_complete = True
        self.fitness_by_f_score = pd.DataFrame()
        self.fitness_df = pd.DataFrame()
        self.ising_full_connectivity = 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5'
        self.heisenberg_xxz_small = 'pauliSet_1J2_xJx_d3+pauliSet_1J3_yJy_d3+pauliSet_2J3_xJx_d3+pauliSet_2J3_zJz_d3'
        self.four_site_true_model = 'pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J4_zJz_d4'
        self.three_site_true_model = 'pauliSet_1J2_zJz_d3+pauliSet_1J3_yJy_d3+pauliSet_1J3_zJz_d3+pauliSet_2J3_xJx_d3+pauliSet_2J3_zJz_d3'
        self.true_model = self.four_site_true_model
        self.true_model = qmla.database_framework.alph(self.true_model)
        self.num_sites = qmla.database_framework.get_num_qubits(self.true_model)
        self.num_probes = 50
        self.max_num_qubits = 7

        self.qhl_models = [
            'pauliSet_1J2_zJz_d3+pauliSet_1J3_yJy_d3+pauliSet_1J3_zJz_d3+pauliSet_2J3_xJx_d3+pauliSet_2J3_zJz_d3',
            'pauliSet_1J3_yJy_d3+pauliSet_1J3_zJz_d3+pauliSet_2J3_xJx_d3+pauliSet_2J3_zJz_d3',
            'pauliSet_1J2_zJz_d3+pauliSet_1J3_zJz_d3+pauliSet_2J3_xJx_d3+pauliSet_2J3_zJz_d3',
        ]
        if self.num_sites < 4 : 
            # to keep state spaces reasonable for development.
            self.base_terms = [
                'x', 'y',  'z'
            ]
        else: 
            self.base_terms = [
                'x', 'z',
            ]
        self.spawn_step = 0

        self.mutation_probability = 0.1

        self.genetic_algorithm = qmla.growth_rules.genetic_algorithms.genetic_algorithm.GeneticAlgorithmQMLA(
            num_sites=self.num_sites,
            true_model = self.true_model,
            base_terms=self.base_terms,
            mutation_probability=self.mutation_probability,
            log_file=self.log_file
        )
        self.true_chromosome = self.genetic_algorithm.true_chromosome
        self.true_chromosome_string = self.genetic_algorithm.true_chromosome_string

        self.num_possible_models = 2**len(self.true_chromosome)

        # self.true_model = 'pauliSet_xJx_1J2_d3+pauliSet_yJy_1J2_d3'
        self.max_num_probe_qubits = self.num_sites
        # default test - 32 generations x 16 starters
        self.max_spawn_depth = 24
        self.initial_num_models = 16
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
        self.hamming_distance_by_generation_step = {
            0: [
                hamming_distance(
                    self.true_chromosome_string,
                    self.genetic_algorithm.chromosome_string(
                        self.genetic_algorithm.map_model_to_chromosome(
                            mod
                        )
                    )
                )
                for mod in self.initial_models
            ]
        }
        self.model_points_at_step = {}     
        self.generation_model_rankings = {} 
        self.models_ranked_by_fitness = {}

        self.tree_completed_initially = False
        self.max_num_models_by_shape = {
            self.num_sites : (self.initial_num_models * self.max_spawn_depth)/10,
            'other': 0
        }
        self.num_processes_to_parallelise_over = 16

        self.max_time_to_consider = 15
        self.min_param = 0.35
        self.max_param = 0.65

    def nominate_champions(self):
        # choose model with highest fitness on final generation
        self.champion_model = self.models_ranked_by_fitness[self.spawn_step][0]

        self.log_print([
            "number mutations:", self.genetic_algorithm.mutation_count, 
            "\nModel rankings on final generation:",
            self.models_ranked_by_fitness[self.spawn_step],
            "\nChampion:", self.champion_model
        ])
        
        return [self.champion_model]


    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        self.spawn_step += 1
        self.log_print([
            "Spawn step:", self.spawn_step
        ])
        model_points = kwargs['branch_model_points']
        self.model_points_at_step[self.spawn_step] = model_points
        evaluation_log_likelihoods = kwargs['evaluation_log_likelihoods']
        model_names_ids = kwargs['model_names_ids']
        sum_wins = sum(list(model_points.values()))
        model_ids = list(model_points.keys())

        # dictionaries which can be used as fitnesses:
        model_number_wins = {}
        model_f_scores = {}
        model_hamming_distances = {}
        model_win_ratio = {}
        model_elo_ratings = {}

        # model rankings  by number of wins
        ranked_model_list = sorted(
            model_points,
            key=model_points.get,
            reverse=True
        )
        ranked_models_by_name = [kwargs['model_names_ids'][m] for m in ranked_model_list]
        self.log_print(["Ranked models:", ranked_model_list, "\n Names:", ranked_models_by_name, "\n with fitnesses:", ])

        self.generation_model_rankings[self.spawn_step] = ranked_models_by_name
        rankings = list(range(1, len(ranked_model_list) + 1))
        rankings.reverse()
        num_points = sum(rankings) # number of points to distribute
        model_points_distributed_by_ranking = list(zip(
            ranked_models_by_name, 
            [r/num_points for r in rankings]
        ))
        model_points_distributed_by_ranking = dict(model_points_distributed_by_ranking)

        # model ratings  (Elo ratings)
        precomputed_ratings = self.ratings_class.get_ratings(list(model_points.keys()))
        original_ratings_by_name = {
            kwargs['model_names_ids'][m] : precomputed_ratings[m]
            for m in model_ids
        }
        min_rating = min(original_ratings_by_name.values())
        ratings_by_name = {
            m : original_ratings_by_name[m] - min_rating
            for m in original_ratings_by_name
        }
        sum_ratings = np.sum(list(ratings_by_name.values()))
        model_elo_ratings = {
            m : ratings_by_name[m]/sum_ratings
            for m in ratings_by_name
        }
        # log likelihoods evaluated against test data
        self.log_print(["Eval log likels:", evaluation_log_likelihoods])
        ll_to_score = {
            a : -1/evaluation_log_likelihoods[a]
            for a in evaluation_log_likelihoods
        }
        s = sum(ll_to_score.values())
        for a in ll_to_score:
            ll_to_score[a] /= s

        # sum_log_likelihoods = sum(evaluation_log_likelihoods.values())
        log_likelihoods = {
            model_names_ids[mod] : ll_to_score[mod]
            for mod in evaluation_log_likelihoods
        }
        self.log_print(["Eval log likels:", log_likelihoods])


        # store info on each model for analysis
        for m in model_ids:
            mod = kwargs['model_names_ids'][m]
            model_number_wins[mod] = model_points[m]
            hamming_dist = self.hamming_distance_model_comparison(
                test_model = mod
            ) # for fitness use 1/H
            model_hamming_distances[mod] = (self.genetic_algorithm.num_terms - hamming_dist)/self.genetic_algorithm.num_terms
            model_f_scores[mod] = self.f_score_model_comparison(
                test_model = mod, 
            )
            model_win_ratio[mod] = model_number_wins[mod]/sum_wins

            # store scores for offline analysis
            self.fitness_by_f_score = (
                self.fitness_by_f_score.append(
                    pd.Series(
                    {
                        'generation' : self.spawn_step,
                        'model_win_ratio' : model_win_ratio[mod], 
                        'model_elo_ratings' : model_elo_ratings[mod], 
                        'original_elo_rating' : original_ratings_by_name[mod],
                        'f_score' : model_f_scores[mod],
                        'model_points_distributed_by_ranking' : model_points_distributed_by_ranking[mod], 
                        'model_hamming_distances' : model_hamming_distances[mod], 
                        'log_likelihood' : evaluation_log_likelihoods[m],
                    }), 
                    ignore_index=True
                )
            )

            self.fitness_df = (
                self.fitness_df.append(
                    pd.Series(
                        {
                            'generation' : self.spawn_step,
                            'f_score' : model_f_scores[mod], 
                            'fitness' : model_f_scores[mod], 
                            'fitness_type' : 'f_score',
                            'active_fitness_method' : self.fitness_method=='f_scores',
                        }
                    ),
                    ignore_index=True
                )
            )
            self.fitness_df = (
                self.fitness_df.append(
                    pd.Series(
                        {
                            'generation' : self.spawn_step,
                            'f_score' : model_f_scores[mod], 
                            'fitness' : model_elo_ratings[mod], 
                            'fitness_type' : 'elo_rating',
                            'active_fitness_method' : self.fitness_method=='elo_ratings',
                        }
                    ),
                    ignore_index=True
                )
            )
            self.fitness_df = (
                self.fitness_df.append(
                    pd.Series(
                        {
                            'generation' : self.spawn_step,
                            'f_score' : model_f_scores[mod], 
                            'fitness' : model_win_ratio[mod], 
                            'fitness_type' : 'model_win_ratio',
                            'active_fitness_method' : self.fitness_method=='model_win_ratio',
                        }
                    ),
                    ignore_index=True
                )
            )
            self.fitness_df = (
                self.fitness_df.append(
                    pd.Series(
                        {
                            'generation' : self.spawn_step,
                            'f_score' : model_f_scores[mod], 
                            'fitness' : model_points_distributed_by_ranking[mod], 
                            'fitness_type' : 'ranking',
                            'active_fitness_method' : self.fitness_method=='ranking',
                        }
                    ),
                    ignore_index=True
                )
            )
            self.fitness_df = (
                self.fitness_df.append(
                    pd.Series(
                        {
                            'generation' : self.spawn_step,
                            'f_score' : model_f_scores[mod], 
                            'fitness' : model_hamming_distances[mod], 
                            'fitness_type' : 'model_hamming_distances',
                            'active_fitness_method' : self.fitness_method=='hamming_distances',
                        }
                    ),
                    ignore_index=True
                )
            )
            self.fitness_df = (
                self.fitness_df.append(
                    pd.Series(
                        {
                            'generation' : self.spawn_step,
                            'f_score' : model_f_scores[mod], 
                            'fitness' : log_likelihoods[mod], 
                            'fitness_type' : 'log_likelihoods',
                            'active_fitness_method' : self.fitness_method=='log_likelihoods',
                        }
                    ),
                    ignore_index=True
                )
            )
        
        self.log_print(
            [
                'Generation {} \nModel Win numbers: \n{} \nF-scores: \n{} \nWin ratio:\n{} \nModel Ratings:\n{} \nRanking: \n{} \nlog_likelihoods: \n{}'.format(
                    self.spawn_step,
                    model_number_wins,
                    model_f_scores,
                    model_win_ratio,
                    ratings_by_name, 
                    model_points_distributed_by_ranking,
                    log_likelihoods
                )                
            ]
        )

        # choose the fitness method to use for the genetic algorithm
        if self.fitness_method == 'elo_ratings':
            genetic_algorithm_fitnesses = model_elo_ratings
        elif self.fitness_method == 'f_scores':
            genetic_algorithm_fitnesses = model_f_scores
        elif self.fitness_method == 'hamming_distances': 
            genetic_algorithm_fitnesses = model_hamming_distances
        elif self.fitness_method == 'number_of_wins':
            genetic_algorithm_fitnesses = model_number_wins
        elif self.fitness_method == 'ranking':
            genetic_algorithm_fitnesses = model_points_distributed_by_ranking
        elif self.fitness_method == 'log_likelihoods':
            genetic_algorithm_fitnesses = log_likelihoods
        else:
            self.log_print(["No fitness method selected for genetic algorithm"])

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
        # get models from genetic algorithm
        new_models = self.genetic_algorithm.genetic_algorithm_step(
            model_fitnesses=genetic_algorithm_fitnesses, 
            num_pairs_to_sample=self.initial_num_models / 2 # for every pair, 2 chromosomes proposed
        )

        hamming_distances = [
            hamming_distance(
                self.true_chromosome_string,
                self.genetic_algorithm.chromosome_string(
                    self.genetic_algorithm.map_model_to_chromosome(
                        mod
                    )
                )
            )
            for mod in new_models
        ]
        self.hamming_distance_by_generation_step[
            self.spawn_step] = hamming_distances

        return new_models

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
            qmla.database_framework.get_constituent_names_from_name(target_model)
        )
        terms = [
            self.latex_name(
                term
            )
            for term in
            qmla.database_framework.get_constituent_names_from_name(
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
            self.log_print(
                [
                    "F score from chromosome {} with mod {} not working against true chrom {}".format(
                        mod, chromosome, self.true_chromosome
                    )
                ]
            )
            raise

    def latex_name(
        self,
        name,
        **kwargs
    ):
        # print("[latex name fnc] name:", name)
        core_operators = list(sorted(qmla.database_framework.core_operator_dict.keys()))
        num_sites = qmla.database_framework.get_num_qubits(name)
        p_str = 'P' * num_sites
        p_str = '+'
        separate_terms = name.split(p_str)

        site_connections = {}
        for c in list(itertools.combinations(list(range(num_sites + 1)), 2)):
            site_connections[c] = []

        term_type_markers = ['pauliSet', 'transverse']
        transverse_axis = None
        for term in separate_terms:
            components = term.split('_')
            if 'pauliSet' in components:
                components.remove('pauliSet')

                for l in components:
                    if l[0] == 'd':
                        dim = int(l.replace('d', ''))
                    elif l[0] in core_operators:
                        operators = l.split('J')
                    else:
                        sites = l.split('J')
                sites = tuple([int(a) for a in sites])
                # assumes like-like pauli terms like xx, yy, zz
                op = operators[0]
                site_connections[sites].append(op)
            elif 'transverse' in components:
                components.remove('transverse')
                for l in components:
                    if l[0] == 'd':
                        transverse_dim = int(l.replace('d', ''))
                    elif l in core_operators:
                        transverse_axis = l

        ordered_connections = list(sorted(site_connections.keys()))
        latex_term = ""

        for c in ordered_connections:
            if len(site_connections[c]) > 0:
                this_term = r"\sigma_{"
                this_term += str(c)
                this_term += "}"
                this_term += "^{"
                for t in site_connections[c]:
                    this_term += "{}".format(t)
                this_term += "}"
                latex_term += this_term
        if transverse_axis is not None:
            latex_term += 'T^{}_{}'.format(transverse_axis, transverse_dim)
        latex_term = "${}$".format(latex_term)
        return latex_term


    def growth_rule_finalise(
        self
    ):        
        self.growth_rule_specific_data_to_store['fitness_correlations'] = dict(self.fitness_correlations)
        chromosomes = sorted(list(set(self.genetic_algorithm.previously_considered_chromosomes)))
        dud_chromosome = str('1' +'0'*self.genetic_algorithm.num_terms)
        if dud_chromosome in chromosomes:
            # TODO generatlise - should be  
            self.log_print(
                [
                    "{} in previous chromosomes:\n{}".format(
                        dud_chromosome, 
                        self.genetic_algorithm.previously_considered_chromosomes
                    )
                ]
            )
        chromosome_numbers = sorted([int(c,2) for c in chromosomes])
        self.growth_rule_specific_data_to_store['chromosomes_tested'] = chromosome_numbers
        try:
            f_scores = []
            for c in chromosomes:
                try:
                    f_scores.append(np.round(self.f_score_from_chromosome_string(c), 3) )
                except:
                    self.log_print(
                        [
                            "Could not compute f score for chromosome: {}".format(c)
                        ]
                    )
            self.growth_rule_specific_data_to_store['f_score_tested_models' ] = f_scores
        except:
            self.log_print(
                [
                    "Could not compute f score for chromosome list: {}".format(chromosomes)
                ]
            )
            pass
        self.growth_rule_specific_data_to_store['true_model_chromosome'] = self.true_chromosome_string
        try:
            self.growth_rule_specific_data_to_store['f_score_fitnesses'] = list(zip(
                self.fitness_by_f_score['f_score'],
                self.fitness_by_f_score['model_win_ratio'],
                self.fitness_by_f_score['model_elo_ratings'],
                self.fitness_by_f_score['original_elo_rating'],
                self.fitness_by_f_score['model_points_distributed_by_ranking'],
                self.fitness_by_f_score['log_likelihood']
                # self.fitness_by_f_score['fitness_ratio_rating_win_rate']
            ))
        except:
            # did not enter generate_models
            pass


    def check_tree_completed(
        self,
        spawn_step,
        **kwargs
    ):
        if spawn_step == self.max_spawn_depth:
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

    def growth_rule_specific_plots(
        self,
        save_directory,
        qmla_id=0, 
    ):

        
        self.plot_correlation_fitness_with_f_score(
            save_to_file = os.path.join(
                save_directory, 
                'correlations_bw_fitness_and_f_score{}.png'.format(qmla_id)
            )
        )

        self.plot_fitness_v_fscore_by_generation(
            save_to_file = os.path.join(
                save_directory, 
                'fitness_types_{}.png'.format(qmla_id)
            )
        )
        self.plot_fitness_v_fscore(
            save_to_file = os.path.join(
                save_directory, 
                'fitness_v_fscore_{}.png'.format(qmla_id)
            )
        )
        self.plot_fitness_v_generation(
            save_to_file = os.path.join(
                save_directory, 
                'fitness_v_generation_{}.png'.format(qmla_id)
            )
        )

    def plot_correlation_fitness_with_f_score(
        self,
        save_to_file
    ):
        plt.clf()
        correlations = pd.DataFrame(
            columns = ['Generation', 'Method', 'Correlation']
        )
        fitness_types_to_ignore = ['f_score', 'model_hamming_distances']
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
                    
                    corr = {
                        'Generation' : g,
                        'Method' : t, 
                        'Correlation' : corr
                    }
                    correlations = correlations.append(
                        pd.Series(corr),
                        ignore_index=True
                    )
                
        self.fitness_correlations = correlations
        self.log_print(["fitness correlations:\n", self.fitness_correlations])
        fig, ax = plt.subplots(figsize=(15, 10))
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
        if save_to_file is not None:
            plt.savefig(save_to_file)


    def plot_fitness_v_fscore_by_generation(
        self, save_to_file
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
        plt.savefig(save_to_file)



    def plot_fitness_v_fscore(self, save_to_file):
        plt.clf()
        fig, ax = plt.subplots()
        sns.set(rc={'figure.figsize':(11.7,8.27)})

        cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        sns.scatterplot(
            x='f_score', 
            y='model_elo_ratings', 
            # hue='generation',
            # palette = cmap,
            label='Rating',
            data = self.fitness_by_f_score,
            ax = ax
        )

        sns.scatterplot(
            x='f_score', 
            y='model_win_ratio', 
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
        ax.figure.savefig(save_to_file)


class GeneticTest(
    Genetic
):
    r"""
    Exactly as the genetic growth rule, but small depth to test quickly.

    """

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )
        self.max_spawn_depth = 4
        self.max_num_probe_qubits = self.num_sites
        self.initial_num_models = 6
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
        self.hamming_distance_by_generation_step = {
            0: [
                hamming_distance(
                    self.true_chromosome_string,
                    self.genetic_algorithm.chromosome_string(
                        self.genetic_algorithm.map_model_to_chromosome(
                            mod
                        )
                    )
                )
                for mod in self.initial_models
            ]
        }
        self.tree_completed_initially = False
        self.max_num_models_by_shape = {
            self.num_sites : (self.initial_num_models * self.max_spawn_depth)/10,
            'other': 0
        }
        self.num_processes_to_parallelise_over = self.initial_num_models
 
