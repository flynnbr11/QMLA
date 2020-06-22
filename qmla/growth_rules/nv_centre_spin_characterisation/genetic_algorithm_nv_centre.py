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

import qmla.growth_rules.genetic_algorithms.genetic_growth_rule
from qmla.growth_rules.genetic_algorithms.genetic_growth_rule import Genetic
import qmla.shared_functionality.probe_set_generation
import qmla.shared_functionality.latex_model_names
import qmla.shared_functionality.expectation_values
import qmla.construct_models


class NVCentreGenticAlgorithm(
    Genetic
):
    def __init__(
        self,
        growth_generation_rule,
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule=growth_generation_rule,
            **kwargs
        )

        self.log_print(["Running GA for NV centre."])

        self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.nv_centre_SAT
        self.qinfer_model_class =  qmla.shared_functionality.qinfer_model_interface.QInferNVCentreExperiment
        self.probe_generation_function = qmla.shared_functionality.probe_set_generation.plus_plus_with_phase_difference
        self.simulator_probe_generation_function = self.probe_generation_function
        self.shared_probes = False
        self.max_time_to_consider = 4.24
        self.true_model = 'pauliSet_1_x_d2+pauliSet_1_y_d2+pauliSet_1_z_d2+pauliSet_1J2_zJz_d2'
        available_terms = [
            'xTi', 
            'yTi', 
            'zTi', 
            'xTx', 
            'yTy', 
            'zTz', 
            'xTy', 
            'xTz', 
            'yTz'
        ]
        self.true_model = 'xTi+yTi+zTi+zTz'
        self.num_sites = qmla.construct_models.get_num_qubits(self.true_model)

        self.genetic_algorithm = qmla.growth_rules.genetic_algorithms.genetic_algorithm.GeneticAlgorithmQMLA(
            num_sites=self.num_sites,
            true_model = self.true_model,
            genes = available_terms, 
            mutation_probability=0.25,
            num_protected_elite_models = 2, 
            unchanged_elite_num_generations_cutoff = 5, 
            log_file=self.log_file
        )
        self.initial_models = self.genetic_algorithm.random_initial_models(16)

        self.expectation_value_function = qmla.shared_functionality.expectation_values.n_qubit_hahn_evolution
        self.branch_comparison_strategy = 'optimal_graph'
        self.tree_completed_initially = False
        self.fitness_method =  'f_scores' # 'elo_ratings' 
        self.max_spawn_depth = 16
        if self.tree_completed_initially:
            self.max_spawn_depth = 1
        self.initial_num_models = len(self.initial_models)
        self.max_num_models_by_shape = {
            self.num_sites : (len(self.initial_models) * self.max_spawn_depth) / 8,
            'other': 0
        }
        self.num_processes_to_parallelise_over = 16
        self.timing_insurance_factor = 1
        self.max_time_to_consider = 20 
        self.min_param = 0.4
        self.max_param = 0.6

