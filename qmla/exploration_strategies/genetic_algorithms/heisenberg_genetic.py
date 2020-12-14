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

from qmla.exploration_strategies.genetic_algorithms.genetic_exploration_strategy \
    import GeneticAlgorithmQMLAFullyConnectedLikewisePauliTerms
import qmla.shared_functionality.probe_set_generation
import qmla.construct_models

class HeisenbergGeneticXYZ(
    GeneticAlgorithmQMLAFullyConnectedLikewisePauliTerms
):

    def __init__(
        self,
        exploration_rules,
        true_model = None, 
        **kwargs
    ):
        # Setup target system
        xyz = True # whether to use HeixXYZ model; False gives HeisXXZ
        if true_model is None: 
            if xyz:
                true_model = 'pauliSet_1J2_yJy_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_yJy_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_yJy_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4'
                self.base_terms = [
                    'x', 'y', 'z',
                ]
            else:
                true_model = 'pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_zJz_d4'
                self.base_terms = [
                    'x', 'z',
                ]
            
            true_model = qmla.construct_models.alph(true_model)
        
        super().__init__(
            exploration_rules=exploration_rules,
            true_model = true_model,
            **kwargs
        )
        self.true_model_terms_params = {
            # reasonably interesting dynamics within 0.5 \pm 0.25
            'pauliSet_1J2_yJy_d4': 0.5215104901916923,
            'pauliSet_1J2_zJz_d4': 0.677532102219103,
            'pauliSet_1J3_zJz_d4': 0.2856421361482581,
            'pauliSet_1J4_yJy_d4': 0.2520347900489445,
            'pauliSet_2J3_xJx_d4': 0.2805221884243438,
            'pauliSet_2J3_yJy_d4': 0.6289731115726565,
            'pauliSet_2J4_xJx_d4': 0.3658869278936159,
            'pauliSet_3J4_xJx_d4': 0.464429107089917,
            'pauliSet_3J4_zJz_d4': 0.3901210315999691
        }

        # Logistics
        self.prune_completed_initially = True
        self.prune_complete = True
        self.fitness_by_f_score = pd.DataFrame()
        self.fitness_df = pd.DataFrame()
        self.num_sites = qmla.construct_models.get_num_qubits(self.true_model)

        self.num_probes = 50
        self.max_num_qubits = 7
        self.num_possible_models = 2**len(self.true_chromosome)
        self.max_num_probe_qubits = self.num_sites

        self.qhl_models = [
            self.true_model
        ]
        self.true_param_cov_mtx_widen_factor = 1
        self.check_champion_reducibility = False

        # Genetic algorithm settings
        self.mutation_probability = 0.15
        self.genetic_algorithm.terminate_early_if_top_model_unchanged = True
        self.true_chromosome = self.genetic_algorithm.true_chromosome
        self.genetic_algorithm.unchanged_elite_num_generations_cutoff = 5
        self.genetic_algorithm.selection_truncation_rate = 2/6
        self.true_chromosome_string = self.genetic_algorithm.true_chromosome_string
        
        # WIDTH/DEPTH OF GENETIC ALGORITHM
        self.max_spawn_depth = 16
        self.initial_num_models = 60

        # Get starting population
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )

        # Settings for model search
        self.branch_comparison_strategy = 'optimal_graph'
        self.tree_completed_initially = False
        self.fraction_particles_for_bf = 0.2
        self.fraction_own_experiments_for_bf = 0.2
        self.fraction_opponents_experiments_for_bf = 0.2
        self.iqle_mode = True

        # Parameter learning
        self.max_time_to_consider = 15
        self.min_param = 0.25
        self.max_param = 0.75
        self.force_evaluation = False
        self.max_time_to_consider = 60
        self.iqle_mode = True

        # Timing info for cluster
        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 0.1
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 6,
            'other': 0
        }

class TestHeisenbergGeneticXYZ(
    HeisenbergGeneticXYZ
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

        self.max_spawn_depth = 2
        self.initial_num_models = 8
        self.branch_comparison_strategy = 'optimal_graph' # minimal'

        # Get starting population
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )
