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

from qmla.exploration_strategies.genetic_algorithms.genetic_exploration_strategy import \
    Genetic, hamming_distance, GeneticAlgorithmQMLAFullyConnectedLikewisePauliTerms
import qmla.shared_functionality.probe_set_generation
import qmla.construct_models


class DemoBayesFactorsByFscore(
    GeneticAlgorithmQMLAFullyConnectedLikewisePauliTerms
):
    r"""
    A single generation of a genetic exploration strategy. 

    Test models of varying F-score, compute pairwise Bayes factors
    between all pairs of models for analysis. 
    """

    def __init__(
        self,
        exploration_rules,
        true_model=None,
        **kwargs
    ):
        if true_model is None:
            self.base_terms = [
                'z',
            ]
            true_model = 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5'
            true_model = qmla.construct_models.alph(true_model)
        
        super().__init__(
            exploration_rules=exploration_rules,
            true_model = true_model,
            base_terms = self.base_terms, 
            **kwargs
        )

        self.prune_completed_initially = True
        self.prune_complete = True
        self.fitness_by_f_score = pd.DataFrame()
        self.fitness_df = pd.DataFrame()
        self.num_sites = qmla.construct_models.get_num_qubits(self.true_model)
        self.num_probes = 50
        self.max_num_qubits = self.num_sites
        self.max_num_probe_qubits = self.num_sites

        self.qhl_models = [
            self.true_model
        ]
        self.true_param_cov_mtx_widen_factor = 1

        # Genetic algorithm details
        self.true_chromosome = self.genetic_algorithm.true_chromosome
        self.true_chromosome_string = self.genetic_algorithm.true_chromosome_string
        self.num_possible_models = 2**len(self.true_chromosome)
        self.mutation_probability = 0.25

        self.genetic_algorithm.terminate_early_if_top_model_unchanged = True
        self.branch_comparison_strategy = 'all'
        self.tree_completed_initially = True
        self.fraction_particles_for_bf = 1
        self.fraction_own_experiments_for_bf = 1
        self.fraction_opponents_experiments_for_bf = 1
        self.iqle_mode = True
        
        self.max_time_to_consider = 15
        self.min_param = 0.25
        self.max_param = 0.75

        test_fitness_models = [
            # F=0
            'pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5', 
            # 'pauliSet_2J4_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5',
            # 0.2 <= f < 0.3
            'pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5', # F=0.2
            # 'pauliSet_1J4_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.3 <= f < 0.4
            'pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5',
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5+pauliSet_4J5_zJz_d5',
            # 0.4 <= f < 0.5
            'pauliSet_1J2_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5',
            # 'pauliSet_1J4_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.5 <= f < 0.6
            'pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 'pauliSet_1J3_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.6 <= f < 0.7
            'pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5',
            # 'pauliSet_1J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.7 <= f < 0.8
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5+pauliSet_4J5_zJz_d5',
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5',            
            # 0.8 <= f < 0.9
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5', # F=0.8
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.9 <= f < 1
            'pauliSet_1J2_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5',
            # F = 1
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5' # F=1
        ]

        self.initial_models = list(
            np.random.choice(test_fitness_models, len(test_fitness_models), replace=False)
        )
        self.initial_models = [qmla.construct_models.alph(m) for m in self.initial_models]
        if self.true_model not in self.initial_models:
            rand_idx = self.initial_models.index(np.random.choice(self.initial_models))
            self.initial_models[rand_idx] = self.true_model

        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        
        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 0.75

class DemoFractionalResourcesBayesFactorsByFscore(
    DemoBayesFactorsByFscore
):

    def __init__(
        self,
        exploration_rules,
        true_model = None, 
        **kwargs
    ):

        super().__init__(
            exploration_rules=exploration_rules,
            true_model = true_model,
            **kwargs
        )

        self.fraction_particles_for_bf = 0.2
        self.fraction_own_experiments_for_bf = 0.2
        self.fraction_opponents_experiments_for_bf = 0.2


class DemoBayesFactorsByFscoreEloGraphs(
    DemoBayesFactorsByFscore
):

    def __init__(
        self,
        exploration_rules,
        true_model = None, 
        **kwargs
    ):

        super().__init__(
            exploration_rules=exploration_rules,
            true_model = true_model,
            **kwargs
        )

        self.fraction_particles_for_bf = 0.2
        self.fraction_own_experiments_for_bf = 0.2
        self.fraction_opponents_experiments_for_bf = 0.2
        self.branch_comparison_strategy = 'optimal_graph'

