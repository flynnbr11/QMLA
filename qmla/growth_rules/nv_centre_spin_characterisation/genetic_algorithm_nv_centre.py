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


class NVCentreSimulatedShortDynamicsGenticAlgorithm(
    Genetic
):
    def __init__(
        self,
        growth_generation_rule,
        true_model=None,
        **kwargs
    ):
        # Fundamental set up
        if true_model is None:
            true_model = 'xTi+yTi+zTi+zTz'
        true_model = qmla.construct_models.alph(true_model)
        available_terms = [
            'xTi', 'yTi', 'zTi', 
            'xTx', 'yTy', 'zTz',
            'xTy', 'xTz', 'yTz'
        ]

        super().__init__(
            growth_generation_rule=growth_generation_rule,
            true_model = true_model,
            genes = available_terms,
            **kwargs
        )

        # Model design/learning
        self.true_model_terms_params = {
            'xTi': 0.92450565,
            'yTi': 6.00664336,
            'zTi': 1.65998543,
            'zTz': 0.76546868,
        }
        self.min_param = 0
        self.max_param = 10

        # Modular functions
        self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.nv_centre_SAT
        self.expectation_value_function = qmla.shared_functionality.n_qubit_hahn_evolution_double_time_reverse
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probe_generation_function = self.probe_generation_function
        self.shared_probes = False
        self.num_sites = qmla.construct_models.get_num_qubits(self.true_model)
        self.expectation_value_function = qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution
        self.model_heuristic_function = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic

        # Genetic algorithm options
        self.tree_completed_initially = False
        # self.branch_comparison_strategy = 'optimal_graph'
        self.fitness_method =  'elo_ratings'  # 'f_score'

        num_models_per_generation = 14
        self.initial_models = self.genetic_algorithm.random_initial_models(num_models_per_generation)
        # self.initial_models = [
        #     'xTi+yTi+zTi',
        #     'xTi+yTi+zTi+zTz', 
        #     'xTi+yTi+zTi+yTy+zTz+yTz', 
        #     'xTi+yTi+zTi+xTx+yTy+zTz+xTy+xTz+yTz', 
        # ]
        self.initial_models = [ 
            qmla.construct_models.alph(m) for m in self.initial_models
        ]
        self.max_spawn_depth = 14

        # Logistics
        self.fraction_particles_for_bf = 0.5
        self.fraction_own_experiments_for_bf = 0.5
        self.fraction_opponents_experiments_for_bf = 0
        self.max_time_to_consider = 4.24
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




class NVCentreExperimentalShortDynamicsGenticAlgorithm(
    NVCentreSimulatedShortDynamicsGenticAlgorithm
):
    def __init__(
        self,
        growth_generation_rule,
        true_model=None,
        **kwargs
    ):

        super().__init__(
            growth_generation_rule=growth_generation_rule,
            true_model = true_model,
            **kwargs
        )
        self.qinfer_model_class =  qmla.shared_functionality.qinfer_model_interface.QInferNVCentreExperiment


    def get_true_parameters(
        self,
    ):        
        self.fixed_true_terms = True
        self.true_hamiltonian = None
        self.true_params_dict = {}
        self.true_params_list = []


    def get_measurements_by_time(
        self
    ):
        data_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'data/NVB_rescale_dataset.p'
            )
        )
        self.log_print([
            "Getting experimental data from {}".format(data_path)
        ])
        self.measurements = pickle.load(
            open(
                data_path,
                'rb'
            )
        )
        return self.measurements