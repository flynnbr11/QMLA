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
from qmla.growth_rules.genetic_algorithms.ising_genetic import IsingGenetic, HeisenbergGeneticXXZ
import qmla.shared_functionality.probe_set_generation
import qmla.construct_models


class ObjectiveFncTestHeisXXZ(
    # IsingGenetic
    HeisenbergGeneticXXZ
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
            # 14 randomly selected models, sorted by increasing F-score.
            'pauliSet_1J3_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4',
            'pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4',
            'pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_zJz_d4',
            'pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4',
            'pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4',
            'pauliSet_1J2_xJx_d4+pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_zJz_d4',
            'pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4',
            'pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_xJx_d4',
            'pauliSet_1J4_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J4_xJx_d4+pauliSet_3J4_zJz_d4',
            'pauliSet_1J2_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4',
            'pauliSet_1J2_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4',
            'pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_xJx_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_xJx_d4+pauliSet_3J4_zJz_d4',
            'pauliSet_1J3_xJx_d4+pauliSet_1J3_zJz_d4+pauliSet_1J4_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_2J3_zJz_d4+pauliSet_2J4_xJx_d4+pauliSet_2J4_zJz_d4+pauliSet_3J4_zJz_d4',
            'pauliSet_1J2_xJx_d4+pauliSet_1J2_zJz_d4+pauliSet_1J3_zJz_d4+pauliSet_2J3_xJx_d4+pauliSet_3J4_zJz_d4'
        ]


class ObjectiveFncTestIsing(
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

        self.test_fitness_models =[
            'pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5',
            'pauliSet_1J4_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_4J5_zJz_d5',
            'pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5',
            'pauliSet_1J4_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J5_zJz_d5+pauliSet_4J5_zJz_d5',
            'pauliSet_1J2_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5',
            'pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_3J5_zJz_d5',
            'pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5+pauliSet_4J5_zJz_d5',
            'pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_2J3_zJz_d5',
            'pauliSet_1J2_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J4_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5',
            'pauliSet_1J3_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_3J5_zJz_d5',
            'pauliSet_1J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_3J4_zJz_d5+pauliSet_3J5_zJz_d5+pauliSet_4J5_zJz_d5',
            'pauliSet_1J2_zJz_d5+pauliSet_1J3_zJz_d5+pauliSet_1J5_zJz_d5+pauliSet_2J5_zJz_d5',
            'pauliSet_1J2_zJz_d5+pauliSet_1J4_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_3J5_zJz_d5',
            'pauliSet_1J2_zJz_d5+pauliSet_2J3_zJz_d5+pauliSet_2J5_zJz_d5+pauliSet_4J5_zJz_d5'
        ]
        self.true_model_terms_params = {
            'pauliSet_1J2_zJz_d5' : 0.44,
            'pauliSet_1J3_zJz_d5' : 0.68,
            'pauliSet_2J3_zJz_d5' : 0.57,
            'pauliSet_2J5_zJz_d5' : 0.35,
            'pauliSet_3J5_zJz_d5' : 0.4
        }


class GenAlgObjectiveFncTest(
    # switch between these parent classes to change target to test obj fncs against
    ObjectiveFncTestHeisXXZ
    # ObjectiveFncTestIsing 
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

        num_models = len(self.test_fitness_models)
        self.initial_models = list(np.random.choice(self.test_fitness_models, num_models, replace=False))
        self.log_print(["Num models for GA: ", len(self.initial_models)])

        self.branch_comparison_strategy = 'minimal' # 'optimal_graph' # 'all' 
        self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        self.hypothetical_final_generation = True
        self.max_time_to_consider = 50
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
        # self.fitness_method = 'akaike_weight'
        self.fitness_method = 'aicc_sq' # 'akaike_info_criterion'
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
        self.fitness_method = 'bic_sq'  # 'bayes_weight' # 
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
        self.max_spawn_depth = 32
        self.initial_num_models = 28

        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )


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

        # pickle some elo graphs
        # num_models = 28
        # # num_models = 14
        # self.initial_models = self.genetic_algorithm.random_initial_models(num_models)

        self.fitness_method = 'elo_rating'
        self.branch_comparison_strategy = 'optimal_graph' # 'all'
        self.force_evaluation = False
        self.exclude_evaluation = True
        self.fraction_particles_for_bf = 0.2
        self.fraction_opponents_experiments_for_bf = 0.2
        self.fraction_own_experiments_for_bf = self.fraction_opponents_experiments_for_bf


