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

from qmla.exploration_strategies.genetic_algorithms.genetic_exploration_strategy import (
    GeneticAlgorithmQMLAFullyConnectedLikewisePauliTerms,
)
import qmla.shared_functionality.probe_set_generation
import qmla.model_building_utilities


class HeisenbergGeneticXYZ(GeneticAlgorithmQMLAFullyConnectedLikewisePauliTerms):
    def __init__(self, exploration_rules, true_model=None, **kwargs):
        # Setup target system
        xyz = True  # whether to use HeixXYZ model; False gives HeisXXZ
        if true_model is None:
            if xyz:
                true_model = "pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4"
                self.base_terms = [
                    "x",
                    "y",
                    "z",
                ]
            else:
                true_model = "pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_zJz_d4"
                self.base_terms = [
                    "x",
                    "z",
                ]

            true_model = qmla.model_building_utilities.alph(true_model)

        super().__init__(
            exploration_rules=exploration_rules, true_model=true_model, **kwargs
        )
        self.true_model_terms_params = {
            # reasonably interesting dynamics within 0.5 \pm 0.25
            "pauliSet_1J2_yJy_d4": 0.5215104901916923,
            "pauliSet_1J2_zJz_d4": 0.677532102219103,
            "pauliSet_1J3_zJz_d4": 0.2856421361482581,
            "pauliSet_1J4_yJy_d4": 0.2520347900489445,
            "pauliSet_2J3_xJx_d4": 0.2805221884243438,
            "pauliSet_2J3_yJy_d4": 0.6289731115726565,
            "pauliSet_2J4_xJx_d4": 0.3658869278936159,
            "pauliSet_3J4_xJx_d4": 0.464429107089917,
            "pauliSet_3J4_zJz_d4": 0.3901210315999691,
        }

        # Logistics
        self.prune_completed_initially = True
        self.prune_complete = True
        self.fitness_by_f_score = pd.DataFrame()
        self.fitness_df = pd.DataFrame()
        self.num_sites = qmla.model_building_utilities.get_num_qubits(self.true_model)

        self.num_probes = 50
        self.max_num_qubits = self.num_sites
        self.num_possible_models = 2 ** len(self.true_chromosome)
        self.max_num_probe_qubits = self.num_sites

        self.qhl_models = [self.true_model]
        self.true_param_cov_mtx_widen_factor = 1
        self.check_champion_reducibility = False

        # Genetic algorithm settings
        self.mutation_probability = 0.15
        self.genetic_algorithm.terminate_early_if_top_model_unchanged = True
        self.true_chromosome = self.genetic_algorithm.true_chromosome
        self.genetic_algorithm.unchanged_elite_num_generations_cutoff = 5
        self.genetic_algorithm.selection_truncation_rate = 2 / 6
        self.true_chromosome_string = self.genetic_algorithm.true_chromosome_string

        # WIDTH/DEPTH OF GENETIC ALGORITHM
        self.max_spawn_depth = 5  # 16
        self.initial_num_models = 60

        # Get starting population
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )

        # Settings for model search
        self.branch_comparison_strategy = "optimal_graph"
        self.tree_completed_initially = False
        self.fraction_particles_for_bf = 0.4
        self.fraction_own_experiments_for_bf = 0.4
        self.fraction_opponents_experiments_for_bf = 0.4

        # Parameter learning
        self.max_time_to_consider = 25
        self.min_param = 0.25
        self.max_param = 0.75
        self.force_evaluation = False
        self.iqle_mode = True

        # Timing info for cluster
        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 0.2
        self.max_num_models_by_shape = {
            self.num_sites: (len(self.initial_models) * self.max_spawn_depth) / 6,
            "other": 0,
        }


class TestHeisenbergGeneticXYZ(HeisenbergGeneticXYZ):
    def __init__(self, exploration_rules, true_model=None, **kwargs):

        super().__init__(
            exploration_rules=exploration_rules, true_model=true_model, **kwargs
        )

        self.max_spawn_depth = 2
        self.initial_num_models = 8
        self.branch_comparison_strategy = "optimal_graph"  # minimal'

        # Get starting population
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )


class HeisenbergGeneticSingleLayer(HeisenbergGeneticXYZ):
    def __init__(
        self, exploration_rules, true_model=None, **kwargs  # using that set by parent
    ):
        true_model = "pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5"
        super().__init__(
            exploration_rules=exploration_rules, true_model=true_model, **kwargs
        )

        test_fitness_models = [
            # TODO these models F1-score is with respect to Ising model - find equivalent for Heis
            # F=0
            "pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5",
            # 'pauliSet_2J4_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5',
            # 0.2 <= f < 0.3
            "pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5",  # F=0.2
            # 'pauliSet_1J4_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.3 <= f < 0.4
            "pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5",
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5+pauliSet_4J5_zJz_d5',
            # 0.4 <= f < 0.5
            "pauliSet_1J2_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5",
            # 'pauliSet_1J4_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.5 <= f < 0.6
            "pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5",
            # 'pauliSet_1J3_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.6 <= f < 0.7
            "pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5",
            # 'pauliSet_1J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.7 <= f < 0.8
            "pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5+pauliSet_4J5_zJz_d5",
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5',
            # 0.8 <= f < 0.9
            "pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5",  # F=0.8
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.9 <= f < 1
            "pauliSet_1J2_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5",
            # F = 1
            "pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5",  # F=1
        ]
        # 10 models per layer, fully connected -> 45 comparisons using optimal_graph; 90 using all

        self.initial_models = list(
            np.random.choice(
                test_fitness_models,
                4,  # len(test_fitness_models), # TODO restore full list
                replace=False,
            )
        )
        if self.true_model not in self.initial_models:
            rand_idx = self.initial_models.index(np.random.choice(self.initial_models))
            self.initial_models[rand_idx] = self.true_model

        self.branch_comparison_strategy = "all"
        self.tree_completed_initially = True
        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites: (len(self.initial_models) * self.max_spawn_depth) / 8,
            "other": 0,
        }

        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 0.75


class HeisenbergGeneticTest(HeisenbergGeneticXYZ):
    def __init__(
        self, exploration_rules, true_model=None, **kwargs  # using that set by parent
    ):
        true_model = "pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5"
        super().__init__(
            exploration_rules=exploration_rules, true_model=true_model, **kwargs
        )
        self.iqle_mode = True


class HeisenbergTestDynamicsReproduction(HeisenbergGeneticXYZ):
    def __init__(
        self, exploration_rules, true_model=None, **kwargs  # using that set by parent
    ):
        true_model = "pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4"
        super().__init__(
            exploration_rules=exploration_rules, true_model=true_model, **kwargs
        )

        test_fitness_models = [
            "pauliSet_1J2_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J3_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_xJx_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J3_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J4_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4",
            "pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J4_xJx_d4+pauliSet_2J4_xJx_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J4_xJx_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J3_xJx_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J4_xJx_d4+pauliSet_2J3_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_yJy_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4",
            "pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J3_zJz_d4",
            "pauliSet_1J2_yJy_d4+pauliSet_1J3_xJx_d4+pauliSet_2J3_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J3_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_yJy_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_yJy_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4",
            "pauliSet_1J2_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_yJy_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_yJy_d4+pauliSet_1J3_zJz_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_yJy_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_yJy_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_yJy_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_yJy_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4",
            "pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J3_yJy_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4",
            "pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_yJy_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_xJx_d4+pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4",
            "pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4",
        ]

        # self.initial_models = list(
        #     np.random.choice(
        #         test_fitness_models,
        #         len(test_fitness_models),  # TODO restore full list
        #         replace=False,
        #     )
        # )
        self.initial_models = [
            qmla.model_building_utilities.alph(m) for m in self.initial_models
        ]
        if self.true_model not in self.initial_models:
            rand_idx = self.initial_models.index(np.random.choice(self.initial_models))
            self.initial_models[rand_idx] = self.true_model
        self.qhl_models = self.initial_models

        self.branch_comparison_strategy = "optimal_graph"  # "all"
        self.fraction_particles_for_bf = 0.01  # don't care about comparisons
        self.fraction_own_experiments_for_bf = 0.01
        self.fraction_opponents_experiments_for_bf = 0.01
        self.tree_completed_initially = True
        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites: (len(self.initial_models) * self.max_spawn_depth) / 8,
            "other": 0,
        }

        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 0.75
