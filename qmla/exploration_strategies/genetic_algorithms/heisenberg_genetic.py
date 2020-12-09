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

from qmla.exploration_strategies.genetic_algorithms.ising_genetic import IsingGenetic
import qmla.shared_functionality.probe_set_generation
import qmla.construct_models

class HeisenbergGeneticXYZ(
    IsingGenetic
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

        # Elo fitness function settings
        self.fitness_method = 'elo_rating'
        self.branch_comparison_strategy = 'optimal_graph'
        self.force_evaluation = False
        self.fraction_particles_for_bf = 0.2
        self.fraction_opponents_experiments_for_bf = 0.2
        self.fraction_own_experiments_for_bf = 0.2
        self.timing_insurance_factor = 0.1

        self.max_time_to_consider = 60
        self.iqle_mode = True
        self.max_spawn_depth = 2 # 32 
        self.initial_num_models =  60 # 28
        self.initial_models = self.genetic_algorithm.random_initial_models(
            num_models=self.initial_num_models
        )

        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 3,
            'other': 0
        }