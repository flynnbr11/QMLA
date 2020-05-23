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
        self.ratings_class = qmla.growth_rules.rating_system.ELORating(
            initial_rating=1500,
            k_const=30
        ) # for use when ranking/rating models
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

        self.mutation_probability = 0.25
        self.genetic_algorithm = qmla.growth_rules.genetic_algorithms.genetic_algorithm.GeneticAlgorithmQMLA(
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

        self.fraction_particles_for_bf = 1
        self.max_num_probe_qubits = self.num_sites
        # default test - 32 generations x 16 starters
        self.fitness_method =  'f_scores' # 'model_win_ratio' # 'ranking' # 'f_scores' # 'hamming_distances'  #'elo_ratings' # 'ranking'
        self.genetic_algorithm.terminate_early_if_top_model_unchanged = True
        self.max_spawn_depth = 16
        self.initial_num_models = 7
        test_fitness_models = [
            # 'pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5',
            # 'pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_4J5_zJz_d5',
            # 'pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_3J4_zJz_d5',
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_2J4_zJz_d5',
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_3J4_zJz_d5',
            # 'pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5',
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_3J4_zJz_d5',
            # 'pauliSet_1J2_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5',
            # 'pauliSet_1J3_zJz_d5+pauliSet_2J5_zJz_d5',
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_4J5_zJz_d5',
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5',
            # 'pauliSet_1J2_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5',
            # 'pauliSet_1J3_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5',
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5'
        ]
        # self.initial_models = test_fitness_models

        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
        # test force true model to appear in first generation
        # if self.true_model not in self.initial_models:
        #     self.initial_models[-1] = self.true_model

        self.tree_completed_initially = False
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth),
            'other': 0
        }
        self.num_processes_to_parallelise_over = 2*len(self.initial_models) + 1

        self.max_time_to_consider = 15
        self.min_param = 0.52
        self.max_param = 0.48
        self.timing_insurance_factor = 3
