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

from qmla.growth_rules import growth_rule
import qmla.shared_functionality.probe_set_generation
import qmla.database_framework

import qmla.growth_rules.genetic_algorithm

__all__ = [
    'Genetic', 
    'GeneticTest'
    # 'GeneticAlgorithmQMLA'
]
# flatten list of lists
def flatten(l): return [item for sublist in l for item in sublist]


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

        self.mutation_probability = 0.1

        self.genetic_algorithm = qmla.growth_rules.genetic_algorithm.GeneticAlgorithmQMLA(
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
        self.fitness_at_step = {}      

        self.tree_completed_initially = False
        self.max_num_models_by_shape = {
            self.num_sites : (self.initial_num_models * self.max_spawn_depth)/10,
            'other': 0
        }
        self.num_processes_to_parallelise_over = 16

        self.max_time_to_consider = 15
        self.min_param = 0.35
        self.max_param = 0.65

    def generate_models(
        self,
        model_list,
        **kwargs
    ):
        # print("[Genetic] Calling generate_models")
        self.log_print(
            [
                "Spawn step:", kwargs['spawn_step']
            ]
        )
        model_points = kwargs['branch_model_points']
        evaluation_log_likelihoods = kwargs['evaluation_log_likelihoods']
        # print("Model points:", model_points)
        # print("kwargs: ", kwargs)
        self.fitness_at_step[kwargs['spawn_step']] = model_points
        model_number_wins = {}
        model_f_scores = {}
        # fitness_by_f_score = {}
        fitness_track = {}

        sum_fitnesses = sum(list(model_points.values()))
        model_ids = list(model_points.keys())
        model_ratings = self.ratings_class.get_ratings(list(model_points.keys()))
        original_ratings_by_name = {
            kwargs['model_names_ids'][m] : model_ratings[m]
            for m in model_ids
        }
        ranked_model_list = sorted(
            original_ratings_by_name,
            key=original_ratings_by_name.get,
            reverse=True
        )
        num_mods = len(ranked_model_list)
        rankings = list(range(1, num_mods + 1))
        rankings.reverse()
        num_points = sum(rankings)
        fitness_by_ranking = list(zip(
            ranked_model_list, 
            [r/num_points for r in rankings]
        ))
        self.log_print(["fitness by ranking:", fitness_by_ranking])
        fitness_by_ranking = dict(fitness_by_ranking)

        min_rating = min(original_ratings_by_name.values())
        ratings_by_name = {
            m : original_ratings_by_name[m] - min_rating
            for m in original_ratings_by_name
        }
        sum_ratings = np.sum(list(ratings_by_name.values()))
        self.log_print(
            [
                "Sum fitnesses:", sum_fitnesses,
                "\nSum ratings:", sum_ratings,
                "\nMin rating:", min_rating
            ]
        )

        ratings_weights = {
            m : ratings_by_name[m]/sum_ratings
            for m in ratings_by_name
        }

        for m in model_ids:
            mod = kwargs['model_names_ids'][m]
            # ratings_by_name[mod] = model_ratings[m]
            model_number_wins[mod] = model_points[m]
            f_score = self.f_score_model_comparison(
                test_model = mod, 
            )
            model_f_scores[mod] = f_score
            fitness_track[mod] = model_number_wins[mod]/sum_fitnesses
            if fitness_track[mod]==0 or ratings_weights[mod] == 0:
                fitness_ratio = None 
            else: 
                fitness_ratio = ratings_weights[mod]/fitness_track[mod]


            self.fitness_by_f_score = (
                self.fitness_by_f_score.append(
                    pd.Series(
                    {
                        'fitness_by_win_ratio' : fitness_track[mod], 
                        'fitness_by_rating' : ratings_weights[mod], 
                        'original_rating' : original_ratings_by_name[mod],
                        'generation' : kwargs['spawn_step'],
                        'f_score' : f_score,
                        'fitness_by_ranking' : fitness_by_ranking[mod], 
                        'log_likelihood' : evaluation_log_likelihoods[m],
                        # 'fitness_ratio_rating_win_rate' : fitness_ratio
                    }), 
                    ignore_index=True
                )
            )

            self.fitness_df = (
                self.fitness_df.append(
                    pd.Series(
                        {
                            'f_score' : f_score, 
                            'fitness' : ratings_weights[mod], 
                            'fitness_type' : 'elo_rating'
                        }
                    ),
                    ignore_index=True
                )
            )
            self.fitness_df = (
                self.fitness_df.append(
                    pd.Series(
                        {
                            'f_score' : f_score, 
                            'fitness' : fitness_track[mod], 
                            'fitness_type' : 'win_ratio'
                        }
                    ),
                    ignore_index=True
                )
            )
            self.fitness_df = (
                self.fitness_df.append(
                    pd.Series(
                        {
                            'f_score' : f_score, 
                            'fitness' : fitness_by_ranking[mod], 
                            'fitness_type' : 'ranking'
                        }
                    ),
                    ignore_index=True
                )
            )
        

        self.log_print(
            [
                'Generation {} \nModel Win numbers: \n{} \nF-scores: \n{} \nWin ratio:\n{} \nModel Ratings:\n{} \nRanking: \n{}'.format(
                    kwargs['spawn_step'],
                    model_number_wins,
                    model_f_scores,
                    fitness_track,
                    ratings_by_name, 
                    fitness_by_ranking
                )                
            ]
        )
        # subtracting minimum rating so that model has 0 probability of being used for selection
        min_model_rating = min(ratings_by_name.values())
        for m in ratings_by_name:
            ratings_by_name[m] = ratings_by_name[m] - min_model_rating
        self.log_print(
            ["Re-rated fitnessses:", ratings_by_name]
        )
        # TEST: instead of relative number of wins, use model f score as fitness
        new_models = self.genetic_algorithm.genetic_algorithm_step(
            # model_fitnesses=model_f_scores,
            # model_fitnesses=model_number_wins,
            model_fitnesses=ratings_by_name, 
            # model_fitnesses=fitness_by_ranking,
            # num_pairs_to_sample=self.initial_num_models
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
            kwargs['spawn_step']] = hamming_distances

        return new_models

    def f_score_model_comparison(
        self,
        test_model,
        target_model=None, 
        # growth_class, 
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
        chromosomes = sorted(list(set(self.genetic_algorithm.previously_considered_chromosomes)))
        if '1000000000' in chromosomes: 
            self.log_print(
                [
                    "{} in previous chromosomes:\n{}".format(
                        '1000000000', 
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
        # self.growth_rule_specific_data_to_store['delta_f_scores'] = self.genetic_algorithm.delta_f_by_generation
        try:
            self.growth_rule_specific_data_to_store['f_score_fitnesses'] = list(zip(
                self.fitness_by_f_score['f_score'],
                self.fitness_by_f_score['fitness_by_win_ratio'],
                self.fitness_by_f_score['fitness_by_rating'],
                self.fitness_by_f_score['original_rating'],
                self.fitness_by_f_score['fitness_by_ranking'],
                self.fitness_by_f_score['log_likelihood']
                # self.fitness_by_f_score['fitness_ratio_rating_win_rate']
            ))
        except:
            # did not enter generate_models
            pass
        # self.growth_rule_specific_data_to_store['fitness'] = self.fitness_df


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
                self.genetic_algorithm.genetic_generation
            ]

            self.log_print(
                [
                    "Terminating search early b/c elite model unchanged in {} iterations.".format(
                        self.genetic_algorithm.unchanged_elite_num_generations_cutoff
                    ),
                    "Declaring champion:", self.champion_model
                ]
            )
            # check if elite model hasn't changed in last N generations
            return True
        else:
            return False

    def growth_rule_specific_plots(
        self,
        save_directory,
        qmla_id=0, 
    ):
        self.plot_fitness_v_fscore(
            save_to_file = os.path.join(
                save_directory, 
                'fitness_v_fscore_{}.png'.format(qmla_id)
            )
        )

    def plot_fitness_v_fscore(self, save_to_file):
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.clf()
        fig, ax = plt.subplots()
        sns.set(rc={'figure.figsize':(11.7,8.27)})

        cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        # bplot = 
        sns.scatterplot(
            x='f_score', 
            y='fitness_by_rating', 
            # hue='generation',
            # palette = cmap,
            label='Rating',
            data = self.fitness_by_f_score,
            ax = ax
        )

        sns.scatterplot(
            x='f_score', 
            y='fitness_by_win_ratio', 
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
        ax.set_xlim((0,1))
        ax.figure.savefig(save_to_file)


def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


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
        self.max_spawn_depth = 2
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
 
