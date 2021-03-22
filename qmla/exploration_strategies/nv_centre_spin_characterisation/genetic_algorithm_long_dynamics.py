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
import pickle

import qmla.exploration_strategies.genetic_algorithms.genetic_exploration_strategy
from qmla.exploration_strategies.genetic_algorithms.genetic_exploration_strategy import Genetic
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
import qmla.shared_functionality.expectation_value_functions
import qmla.model_building_utilities



class NVCentreSimulatedLongDynamicsGenticAlgorithm(
    Genetic
):
    def __init__(
        self,
        exploration_rules,
        true_model=None,
        **kwargs
    ):

        # Fundamental set up
        # if true_model is None:
        #     true_model = 'pauliSet_1J2_zJz_d2+pauliSet_1_z_d2+pauliSet_2_x_d2+pauliSet_2_y_d2+pauliSet_2_z_d2'
        # true_model = qmla.model_building_utilities.alph(true_model)
        self._set_true_params()
        self.true_model = '+'.join(
            (self.true_model_terms_params.keys())
        )
        self.true_model = qmla.model_building_utilities.alph(self.true_model)
        available_terms = [
            'pauliSet_1_x_d2',
            'pauliSet_1_y_d2',
            'pauliSet_1_z_d2',
            'pauliSet_2_x_d2',
            'pauliSet_2_y_d2',
            'pauliSet_2_z_d2',
            'pauliSet_1J2_xJx_d2',
            'pauliSet_1J2_yJy_d2',
            'pauliSet_1J2_zJz_d2',
        ]

        super().__init__(
            exploration_rules=exploration_rules,
            true_model = self.true_model,
            genes = available_terms,
            **kwargs
        )

        self._set_true_params() # again in case over written by parent __init__

        # Modular functions
        # self.latex_string_map_subroutine = qmla.shared_functionality.latex_model_names.nv_centre_SAT
        self.system_probes_generation_subroutine = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probes_generation_subroutine = self.system_probes_generation_subroutine
        self.shared_probes = True
        self.num_sites = qmla.model_building_utilities.get_num_qubits(self.true_model)
        self.expectation_value_subroutine = qmla.shared_functionality.expectation_value_functions.n_qubit_hahn_evolution
        self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.TimeList
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.SampleOrderMagnitude
        # self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic

        # Genetic algorithm options
        self.tree_completed_initially = False
        self.branch_comparison_strategy = 'optimal_graph'
        self.fitness_method =  'elo_rating'

        num_models_per_generation = 14
        self.initial_models = self.genetic_algorithm.random_initial_models(num_models_per_generation)
        # self.initial_models = [
        #     'xTi+yTi+zTi',
        #     'xTi+yTi+zTi+zTz', 
        #     'xTi+yTi+zTi+yTy+zTz+yTz', 
        #     'xTi+yTi+zTi+xTx+yTy+zTz+xTy+xTz+yTz', 
        # ]
        self.initial_models = [ 
            qmla.model_building_utilities.alph(m) for m in self.initial_models
        ]
        self.max_spawn_depth = 12

        # Logistics
        self.fraction_particles_for_bf = 0.25
        self.fraction_own_experiments_for_bf = 0.5
        self.fraction_opponents_experiments_for_bf = 0.5
        # self.max_time_to_consider = 10
        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        # self.num_processes_to_parallelise_over = 16
        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 0.35

    def _set_true_params(self):

        # set target model
        self._setup_true_model_2_qubit_approx()

        self.true_model = '+'.join(
            (self.true_model_terms_params.keys())
        )
        self.true_model = qmla.model_building_utilities.alph(self.true_model)
        self.availalbe_pauli_terms  = ['x', 'y', 'z']

        self.max_time_to_consider = 100e-6
        self.plot_time_increment = 0.5e-6

        self.max_num_qubits = 5
        test_prior_info = {}      
        paulis_to_include = self.availalbe_pauli_terms

        for pauli in paulis_to_include:

            for num_qubits in range(1, 1+self.max_num_qubits):
        
                spin_rotation_term = 'pauliSet_1_{p}_d{N}'.format(
                    p=pauli, N=num_qubits)
                test_prior_info[spin_rotation_term] = (5e9, 2e9)

                for j in range(2, 1+num_qubits):

                    nuclei_rotation = 'pauliSet_{j}_{p}_d{N}'.format(
                        j = j, 
                        p = pauli, 
                        N = num_qubits
                    )
                    test_prior_info[nuclei_rotation] = (5e4, 2e4)

                    coupling_w_spin = 'pauliSet_1J{j}_{p}J{p}_d{N}'.format(
                        j = j, 
                        p = pauli,
                        N = num_qubits
                    )
                    test_prior_info[coupling_w_spin] = (5e5, 2e5)

                    # TODO add transverse terms

        self.gaussian_prior_means_and_widths = test_prior_info

    def _setup_true_model_2_qubit_approx(self,):

        n_qubits = 2
        self.true_model_terms_params = {
            # spin
            'pauliSet_1_z_d{}'.format(n_qubits) : 2e9,
            
            # coupling with 2nd qubit
            'pauliSet_1J2_zJz_d{}'.format(n_qubits) : 0.2e6, 
            # 'pauliSet_1J2_yJy_d{}'.format(n_qubits) : 0.4e6, 
            # 'pauliSet_1J2_xJx_d{}'.format(n_qubits) : 0.2e6, 

            # carbon nuclei - 2nd qubit
            'pauliSet_2_x_d{}'.format(n_qubits) : 66e3,
            'pauliSet_2_y_d{}'.format(n_qubits) : 66e3,
            'pauliSet_2_z_d{}'.format(n_qubits) : 15e3,
        }
