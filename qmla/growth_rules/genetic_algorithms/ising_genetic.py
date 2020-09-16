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


class IsingGenetic(
    GeneticAlgorithmQMLAFullyConnectedLikewisePauliTerms
):

    def __init__(
        self,
        growth_generation_rule,
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
            growth_generation_rule=growth_generation_rule,
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
        self.iqle_mode = False

        self.max_num_models_by_shape = {
            self.num_sites : 0.1*(len(self.initial_models) * self.max_spawn_depth),
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


class IsingGeneticTest(
    IsingGenetic
):

    def __init__(
        self,
        growth_generation_rule,
        true_model = None, 
        **kwargs
    ):
        
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            true_model = true_model,
            **kwargs
        )
        self.true_model_terms_params = {
            'pauliSet_1J2_zJz_d5' : 0.44,
            'pauliSet_1J3_zJz_d5' : 0.68,
            'pauliSet_2J3_zJz_d5' : 0.57,
            'pauliSet_2J5_zJz_d5' : 0.35,
            'pauliSet_3J5_zJz_d5' : 0.4
        }

        # test F map for random set of 10 models
        self.initial_models = self.genetic_algorithm.random_initial_models(4)
        self.max_spawn_depth = 2


        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        self.timing_insurance_factor = 0.5

class IsingGeneticSingleLayer(
    IsingGenetic
):

    def __init__(
        self,
        growth_generation_rule,
        true_model = None, # using that set by parent 
        **kwargs
    ):
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            true_model = true_model, 
            **kwargs
        )

        # pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5

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
        # 10 models per layer, fully connected -> 45 comparisons using optimal_graph; 90 using all

        self.initial_models = list(np.random.choice(test_fitness_models, 6, replace=False))
        if self.true_model not in self.initial_models:
            rand_idx = self.initial_models.index(np.random.choice(self.initial_models))
            self.initial_models[rand_idx] = self.true_model

        self.branch_comparison_strategy = 'optimal_graph' # 'all' 
        self.tree_completed_initially = True
        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        
        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 0.75


class HeisenbergGeneticXXZ(
    IsingGenetic
):

    def __init__(
        self,
        growth_generation_rule,
        true_model = None, 
        **kwargs
    ):
        if true_model is None: 
            true_model = 'pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_zJz_d4'
            true_model = qmla.construct_models.alph(true_model)
            self.base_terms = [
                'x', 'z',
            ]
        
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            true_model = true_model,
            **kwargs
        )
        self.true_model_terms_params = {
            'pauliSet_1J2_zJz_d5' : 0.44,
            'pauliSet_1J3_zJz_d5' : 0.68,
            'pauliSet_2J3_zJz_d5' : 0.57,
            'pauliSet_2J5_zJz_d5' : 0.35,
            'pauliSet_3J5_zJz_d5' : 0.4
        }

        # test F map for random set of 10 models
        num_models = 28
        self.initial_models = self.genetic_algorithm.random_initial_models(num_models)
        self.max_spawn_depth = 16
        self.fitness_method = 'elo_rating'
        self.branch_comparison_strategy = 'optimal_graph'
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        self.timing_insurance_factor = 0.5

