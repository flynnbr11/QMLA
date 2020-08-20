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

from qmla.growth_rules.genetic_algorithms.genetic_growth_rule import \
    Genetic, hamming_distance, GeneticAlgorithmQMLAFullyConnectedLikewisePauliTerms
import qmla.shared_functionality.probe_set_generation
import qmla.construct_models


class HeisenbergGenetic(
    # GeneticAlgorithmQMLAFullyConnectedLikewisePauliTerms
    Genetic
):

    def __init__(
        self,
        growth_generation_rule,
        true_model=None,
        **kwargs
    ):
        self.base_terms = [
            'x', 'y', 'z',
        ]
        all_terms = [
            'pauliSet_1J2_xJx_d4', 'pauliSet_1J2_yJy_d4', 'pauliSet_1J2_zJz_d4', # 1,2
            'pauliSet_1J3_xJx_d4', 'pauliSet_1J3_yJy_d4', 'pauliSet_1J3_zJz_d4', # 1,3
            'pauliSet_1J4_xJx_d4', 'pauliSet_1J4_yJy_d4', 'pauliSet_1J4_zJz_d4', # 1,4
            'pauliSet_2J3_xJx_d4', 'pauliSet_2J3_yJy_d4', 'pauliSet_2J3_zJz_d4', # 2,3 
            'pauliSet_2J4_xJx_d4', 'pauliSet_2J4_yJy_d4', 'pauliSet_2J4_zJz_d4', # 2,4
            'pauliSet_3J4_xJx_d4', 'pauliSet_3J4_yJy_d4', 'pauliSet_3J4_zJz_d4', # 3,4
        ]
        true_model = 'pauliSet_1J2_xJx_d4+pauliSet_1J3_yJy_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_yJy_d4+pauliSet_3J4_xJx_d4'
        true_model = qmla.construct_models.alph(true_model)
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            true_model = true_model,
            base_terms = self.base_terms, 
            genes = all_terms, 
            **kwargs
        )

        self.prune_completed_initially = True
        self.prune_complete = True
        self.fitness_by_f_score = pd.DataFrame()
        self.fitness_df = pd.DataFrame()
        self.num_sites = qmla.construct_models.get_num_qubits(self.true_model)
        self.num_probes = 50
        self.max_num_qubits = 7

        self.qhl_models = [
            self.true_model
        ]
        self.true_param_cov_mtx_widen_factor = 1

        self.mutation_probability = 0.25

        self.true_chromosome = self.genetic_algorithm.true_chromosome
        self.true_chromosome_string = self.genetic_algorithm.true_chromosome_string

        self.num_possible_models = 2**len(self.true_chromosome)

        self.max_num_probe_qubits = self.num_sites
        # default test - 32 generations x 16 starters
        self.fitness_method = 'elo_ratings' # 'win_ratio'  # 'number_wins'  # 'ranking' # 'f_score' # 'hamming_distance' #  
        self.genetic_algorithm.terminate_early_if_top_model_unchanged = True
        self.max_spawn_depth = 16
        self.initial_num_models = 14

        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
        # test force true model to appear in first generation
        # if self.true_model not in self.initial_models:
        #     self.initial_models[-1] = self.true_model


        self.branch_comparison_strategy = 'optimal_graph'
        self.tree_completed_initially = False
        self.fraction_particles_for_bf = 0.25
        self.fraction_own_experiments_for_bf = 0.5
        self.fraction_opponents_experiments_for_bf = 0.5
        self.fitness_method = 'elo_rating' 
        self.iqle_mode = False

        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth),
            'other': 0
        }
        self.num_processes_to_parallelise_over = min(2*len(self.initial_models) + 1, 16)
        if not self.tree_completed_initially:
            self.num_processes_to_parallelise_over = 16
        
        self.max_time_to_consider = 15
        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 0.5
        self.min_param = 0.25
        self.max_param = 0.75
