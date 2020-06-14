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

from qmla.growth_rules.genetic_algorithms.genetic_growth_rule import Genetic, hamming_distance
import qmla.shared_functionality.probe_set_generation
import qmla.database_framework


class IsingGenetic(
    Genetic
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
        # self.ratings_class = qmla.growth_rules.rating_system.ELORating(
        #     initial_rating=1500,
        #     k_const=30
        # ) # for use when ranking/rating models
        # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.zero_state_probes
        self.prune_completed_initially = True
        self.prune_complete = True
        self.fitness_by_f_score = pd.DataFrame()
        self.fitness_df = pd.DataFrame()
        self.true_model = 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5'
        # self.true_model = 'pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_zJz_d4' # four sites
        self.true_model = qmla.database_framework.alph(self.true_model)
        self.num_sites = qmla.database_framework.get_num_qubits(self.true_model)
        self.num_probes = 50
        self.max_num_qubits = 7

        self.qhl_models = [
            self.true_model
        ]
        self.base_terms = [
            'z',
        ]
        self.true_param_cov_mtx_widen_factor = 1

        self.mutation_probability = 0.25
        self.genetic_algorithm = qmla.growth_rules.genetic_algorithms.genetic_algorithm.GeneticAlgorithmFullyConnectedLikewisePauliTerms(
            num_sites=self.num_sites,
            true_model = self.true_model,
            base_terms=self.base_terms,
            mutation_probability=self.mutation_probability,
            num_protected_elite_models = 2, 
            unchanged_elite_num_generations_cutoff = 5, 
            log_file=self.log_file
        )

        self.true_chromosome = self.genetic_algorithm.true_chromosome
        self.true_chromosome_string = self.genetic_algorithm.true_chromosome_string

        self.num_possible_models = 2**len(self.true_chromosome)

        self.max_num_probe_qubits = self.num_sites
        # default test - 32 generations x 16 starters
        self.fitness_method =  'win_ratio'  # 'number_wins'  # 'ranking' # 'f_score' # 'hamming_distance' # 'elo_ratings' 
        self.genetic_algorithm.terminate_early_if_top_model_unchanged = True
        self.max_spawn_depth = 16
        self.initial_num_models = 10

        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
        # test force true model to appear in first generation
        # if self.true_model not in self.initial_models:
        #     self.initial_models[-1] = self.true_model

        self.tree_completed_initially = True
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth),
            'other': 0
        }
        self.num_processes_to_parallelise_over = min(2*len(self.initial_models) + 1, 16)
        if not self.tree_completed_initially:
            self.num_processes_to_parallelise_over = 16
        
        self.max_time_to_consider = 15
        self.min_param = 0.45
        self.max_param = 0.55
        self.timing_insurance_factor = 1


class IsingGeneticTest(
    IsingGenetic
):

    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )

        test_fitness_models = [
            # F=0
            'pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5', 
            'pauliSet_2J4_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5',
            # 0.2 <= f < 0.3
            'pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5', # F=0.2
            'pauliSet_1J4_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.3 <= f < 0.4
            'pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5',
            'pauliSet_1J2_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5+pauliSet_4J5_zJz_d5',
            # 0.4 <= f < 0.5
            'pauliSet_1J2_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5',
            'pauliSet_1J4_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.5 <= f < 0.6
            'pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            'pauliSet_1J3_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.6 <= f < 0.7
            'pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5',
            'pauliSet_1J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.7 <= f < 0.8
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5+pauliSet_4J5_zJz_d5',
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5',            
            # 0.8 <= f < 0.9
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5', # F=0.8
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.9 <= f < 1
            'pauliSet_1J2_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5',
            # F = 1
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5' # F=1
        ]
        # self.initial_models = list(np.random.choice(test_fitness_models, 10, replace=False))
        self.initial_models = self.genetic_algorithm.random_initial_models(10)

        # if self.true_model not in self.initial_models:
        #     rand_idx = self.initial_models.index(np.random.choice(self.initial_models))
        #     self.initial_models[rand_idx] = self.true_model

        # test F map for random set of 10 models

        self.branch_comparison_strategy = 'optimal_graph'
        self.tree_completed_initially = False
        self.fraction_particles_for_bf = 0.5
        self.fraction_experiments_for_bf = 0.5
        self.fitness_method =  'elo_ratings' 
        self.max_spawn_depth = 3
        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 0.35
        self.max_time_to_consider = 20 
        self.min_param = 0.4
        self.max_param = 0.6
