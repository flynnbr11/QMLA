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

import qmla.growth_rules.genetic_algorithms.genetic_growth_rule
from qmla.growth_rules.genetic_algorithms.genetic_growth_rule import Genetic
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
import qmla.shared_functionality.expectation_values
import qmla.construct_models



class NVCentreGenticAlgorithmPrelearnedParameters(
    Genetic
):
    def __init__(
        self,
        growth_generation_rule,
        true_model=None,
        **kwargs
    ):

        # Fundamental set up
        # if true_model is None:
        #     true_model = 'pauliSet_1J2_zJz_d2+pauliSet_1_z_d2+pauliSet_2_x_d2+pauliSet_2_y_d2+pauliSet_2_z_d2'
        # true_model = qmla.construct_models.alph(true_model)
        self._set_true_params()
        self.true_model = '+'.join(
            (self.true_model_terms_params.keys())
        )
        self.true_model = qmla.construct_models.alph(self.true_model)
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
            # 'pauliSet_1J2_xJz_d2' # test
        ]

        super().__init__(
            growth_generation_rule=growth_generation_rule,
            true_model = self.true_model,
            genes = available_terms,
            **kwargs
        )

        self._set_true_params() # again in case over written by parent __init__

        # Modular functions
        # self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.nv_centre_SAT
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        # self.probe_generation_function = qmla.shared_functionality.probe_set_generation.separable_probe_dict # doesn't matter here
        self.evaluation_probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        # self.evaluation_probe_generation_function = qmla.shared_functionality.probe_set_generation.separable_probe_dict
        # self.evaluation_probe_generation_function = qmla.shared_functionality.probe_set_generation.tomographic_basis
        self.num_eval_probes = 36
        self.num_eval_points = 100
        self.simulator_probe_generation_function = self.probe_generation_function
        self.shared_probes = True
        self.num_probes = 5
        self.num_sites = qmla.construct_models.get_num_qubits(self.true_model)
        self.expectation_value_function = qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution
        # self.expectation_value_function = qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution_double_time_reverse
        self.expectation_value_function = qmla.shared_functionality.expectation_values.probability_from_default_expectation_value
        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.TimeList
        # self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.SampleOrderMagnitude
        # self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic

        self.iqle_mode = True
        self.qinfer_resampler_a = 1
        self.qinfer_resampler_threshold = 0.0

        self.qhl_models = [
            'pauliSet_1J2_zJz_d2+pauliSet_1_z_d2+pauliSet_2_x_d2+pauliSet_2_y_d2+pauliSet_2_z_d2', # True

            # extra coupling in X,Y
            # 'pauliSet_1J2_xJx_d2+pauliSet_1J2_zJz_d2+pauliSet_1_z_d2+pauliSet_2_x_d2+pauliSet_2_y_d2+pauliSet_2_z_d2', # 1 extra invisible to |+>
            # 'pauliSet_1J2_yJy_d2+pauliSet_1J2_zJz_d2+pauliSet_1_z_d2+pauliSet_2_x_d2+pauliSet_2_y_d2+pauliSet_2_z_d2', # 1 extra invisible to |+>
            'pauliSet_1J2_xJx_d2+pauliSet_1J2_yJy_d2+pauliSet_1J2_zJz_d2+pauliSet_1_z_d2+pauliSet_2_x_d2+pauliSet_2_y_d2+pauliSet_2_z_d2', # 1 extra invisible to |+>

            # extra rotation on spin qubit
            # 'pauliSet_1J2_zJz_d2+pauliSet_1_x_d2+pauliSet_1_z_d2+pauliSet_2_x_d2+pauliSet_2_y_d2+pauliSet_2_z_d2', 
            # 'pauliSet_1J2_zJz_d2+pauliSet_1_y_d2+pauliSet_1_z_d2+pauliSet_2_x_d2+pauliSet_2_y_d2+pauliSet_2_z_d2', 

            # incorrect model
            # 'pauliSet_1_x_d2+pauliSet_1J2_yJy_d2'
        ]

        self.qhl_models = [
            qmla.construct_models.alph(m) for m in self.qhl_models
        ]


        # Genetic algorithm options
        self.tree_completed_initially = False
        self.branch_comparison_strategy = 'minimal' # 'optimal_graph' #'sparse_connection'
        self.fitness_method =  'log_likelihoods' # 'elo_ratings'  # 'f_score'

        num_models_per_generation = 14
        self.max_spawn_depth = 16
        self.initial_models = self.genetic_algorithm.random_initial_models(num_models_per_generation)
        self.initial_models = [ 
            qmla.construct_models.alph(m) for m in self.initial_models
        ]
        if self.tree_completed_initially:
            self.initial_models = self.qhl_models

        # Logistics
        self.force_evaluation = True
        self.fraction_particles_for_bf = 0.1 # BF not meaningful here so minimising cost
        self.fraction_own_experiments_for_bf = 0.1
        self.fraction_opponents_experiments_for_bf = 0
        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        # self.num_processes_to_parallelise_over = 16
        self.num_processes_to_parallelise_over = len(self.initial_models)
        self.timing_insurance_factor = 1

    def _set_true_params(self):

        # set target model
        self._setup_true_model_2_qubit_approx()

        self.true_model = '+'.join(
            (self.true_model_terms_params.keys())
        )
        self.true_model = qmla.construct_models.alph(self.true_model)
        self.availalbe_pauli_terms  = ['x', 'y', 'z']

        self.max_time_to_consider = 100e-6
        self.plot_time_increment = self.max_time_to_consider / 100
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

    def get_prior(self, model_name, **kwargs):
        prior = qmla.shared_functionality.prior_distributions.prelearned_true_parameters_prior(
            model_name = model_name, 
            true_parameters = self.true_model_terms_params, 
            prior_specific_terms=self.gaussian_prior_means_and_widths,
            default_parameter = 0, 
            default_width = 0.05, 
            fraction_true_parameter_width = 0.01,
            log_file = self.log_file, 
            log_identifier= 'PrelearnedPrior'
        )

        return prior

    def generate_evaluation_data(
        self, 
        num_times = 100, 
        **kwargs
    ):
        r"""
        Generates sequential, equally spaced times
        """
        # times = np.random.rand(num_times)
        # min_t = self.max_time_to_consider / int(num_times)
        # delta_t = 10*min_t # effectively how many iterations each time is eventually learned for
        times = np.arange(
            self.plot_time_increment, 
            self.max_time_to_consider, 
            # 10*self.plot_time_increment # to speedup test
            self.plot_time_increment
        )
        eval_data = super().generate_evaluation_data(
            num_probes = self.num_eval_probes, 
            evaluation_times = times, 
            num_eval_points = self.num_eval_points,
            **kwargs
        )

        self.log_print([
            "Evaluation data:", eval_data
        ])

        return eval_data

    def get_evaluation_prior(
        self, 
        model_name, 
        estimated_params, 
        cov_mt, 
        **kwargs
    ):
        posterior_distribution = self.get_prior(model_name = model_name)
        return posterior_distribution
