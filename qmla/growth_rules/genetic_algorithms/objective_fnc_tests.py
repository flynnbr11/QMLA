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
from qmla.growth_rules.genetic_algorithms.ising_genetic import IsingGenetic
import qmla.shared_functionality.probe_set_generation
import qmla.construct_models


class GenAlgObjectiveFncTest(
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

        self.test_fitness_models = [
            # F=0
            # 'pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5', 
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
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5+pauliSet_4J5_zJz_d5',
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5',            
            # 0.8 <= f < 0.9
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5', # F=0.8
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5',
            # 0.9 <= f < 1
            'pauliSet_1J2_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5',
            # F = 1
            # 'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5' # F=1
        ]
        # 10 models per layer, fully connected -> 45 comparisons using optimal_graph; 90 using all

        num_models =  len(self.test_fitness_models) # 4
        self.initial_models = list(np.random.choice(self.test_fitness_models, num_models, replace=False))
        self.log_print(["Num models for GA: ", len(self.initial_models)])
        # if self.true_model not in self.initial_models:
        #     rand_idx = self.initial_models.index(np.random.choice(self.initial_models))
        #     self.initial_models[rand_idx] = self.true_model

        # self.initial_models = self.test_fitness_models # a batch of 14 models to feed every GR
        self.branch_comparison_strategy = 'minimal' # 'optimal_graph' # 'all' 
        self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        self.hypothetical_final_generation = True
        
        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 0.75


class ObjFncLL(GenAlgObjectiveFncTest):
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
        self.fitness_method = 'inverse_ll'
        self.branch_comparison_strategy = 'minimal'
        self.force_evaluation = True
        self.fraction_particles_for_bf = 0.05
        self.fraction_opponents_experiments_for_bf = 0
        self.fraction_own_experiments_for_bf = 0.05

class ObjFncAIC(GenAlgObjectiveFncTest):
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
        self.fitness_method = 'akaike_weight'
        self.branch_comparison_strategy = 'minimal'
        self.force_evaluation = True
        self.fraction_particles_for_bf = 0.05
        self.fraction_opponents_experiments_for_bf = 0
        self.fraction_own_experiments_for_bf = 0.05

class ObjFncBIC(GenAlgObjectiveFncTest):
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
        self.fitness_method = 'bayes_weight'
        self.branch_comparison_strategy = 'minimal'
        self.force_evaluation = True
        self.fraction_particles_for_bf = 0.05
        self.fraction_opponents_experiments_for_bf = 0
        self.fraction_own_experiments_for_bf = 0.05

class ObjFncResiduals(GenAlgObjectiveFncTest):
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
        self.fitness_method = 'rs_mean_sq'
        self.branch_comparison_strategy = 'minimal'
        self.force_evaluation = True
        self.fraction_particles_for_bf = 0.05
        self.fraction_opponents_experiments_for_bf = 0
        self.fraction_own_experiments_for_bf = 0.05


class ObjFncBFP(GenAlgObjectiveFncTest):
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
        self.fitness_method = 'bf_points'
        self.branch_comparison_strategy = 'all'
        self.force_evaluation = False
        self.exclude_evaluation = True
        self.fraction_particles_for_bf = 0.5
        self.fraction_opponents_experiments_for_bf = 0.5
        self.fraction_own_experiments_for_bf = 0.5


class ObjFncRank(GenAlgObjectiveFncTest):
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
        self.fitness_method = 'bf_rank'
        self.branch_comparison_strategy = 'all'
        self.force_evaluation = False
        self.exclude_evaluation = True
        self.fraction_particles_for_bf = 0.5
        self.fraction_opponents_experiments_for_bf = 0.5
        self.fraction_own_experiments_for_bf = 0.5


class ObjFncElo(GenAlgObjectiveFncTest):
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
        self.fitness_method = 'elo_rating'
        self.branch_comparison_strategy = 'optimal_graph' # 'all'
        self.force_evaluation = False
        self.exclude_evaluation = True
        self.fraction_particles_for_bf = 0.5
        self.fraction_opponents_experiments_for_bf = 0.5
        self.fraction_own_experiments_for_bf = 0.5


